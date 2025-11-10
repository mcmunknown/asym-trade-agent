#include "websocket_client.h"
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ssl/stream.hpp>
#include <nlohmann/json.hpp>
#include <chrono>
#include <algorithm>

namespace mathcore {

namespace beast = boost::beast;
namespace http = beast::http;
namespace websocket = beast::websocket;
namespace net = boost::asio;
using tcp = net::ip::tcp;

// WebSocket implementation details
class WebSocketClient::WebSocketImpl {
public:
    WebSocketImpl(net::io_context& io_context, boost::asio::ssl::context& ssl_ctx)
        : resolver_(io_context), ws_(io_context, ssl_ctx), ssl_ctx_(ssl_ctx) {}
    
    bool connect(const std::string& host, const std::string& port) {
        try {
            // Resolve host
            auto const results = resolver_.resolve(host, port);
            
            // Make the connection on the IP address we get from a lookup
            auto ep = net::connect(ws_.next_layer(), results);
            
            host_ = host + ":" + port;
            
            // Perform SSL handshake
            if (!SSL_set_tlsext_host_name(ws_.next_layer().native_handle(),
                                          host.c_str())) {
                return false;
            }
            
            // Perform the websocket handshake
            ws_.handshake(host, "/v5/public/linear");
            
            return true;
        } catch (const std::exception& e) {
            return false;
        }
    }
    
    void disconnect() {
        try {
            ws_.close(websocket::close_code::normal);
        } catch (...) {}
    }
    
    void send_message(const std::string& message) {
        try {
            ws_.write(net::buffer(message));
        } catch (...) {}
    }
    
    void start_reading() {
        ws_.async_read(buffer_,
                    [this](beast::error_code ec, std::size_t bytes_transferred) {
                        if (!ec) {
                            on_message_received();
                            start_reading(); // Continue reading
                        }
                    });
    }
    
    void set_message_callback(std::function<void(const std::string&)> callback) {
        message_callback_ = callback;
    }
    
private:
    void on_message_received() {
        std::string message = beast::buffers_to_string(buffer_.data());
        buffer_.consume(buffer_.size());
        
        if (message_callback_) {
            message_callback_(message);
        }
    }
    
    tcp::resolver resolver_;
    websocket::stream<beast::ssl_stream<tcp::socket>> ws_;
    boost::asio::ssl::context& ssl_ctx_;
    beast::flat_buffer buffer_;
    std::string host_;
    std::function<void(const std::string&)> message_callback_;
};

WebSocketClient::WebSocketClient(const std::string& host, const std::string& port, bool use_ssl)
    : host_(host), port_(port), use_ssl_(use_ssl), connected_(false), reconnecting_(false),
      running_(false), stats_{0, 0, 0, 0, 1.0} {
    
    io_context_ = std::make_unique<net::io_context>();
    
    if (use_ssl) {
        ssl_context_ = std::make_unique<net::ssl::context>(net::ssl::context::tlsv12_client);
        ssl_context_->set_verify_mode(net::ssl::verify_peer);
        ssl_context_->set_default_verify_paths();
    }
}

WebSocketClient::~WebSocketClient() {
    disconnect();
    running_ = false;
    if (io_thread_ && io_thread_->joinable()) {
        io_thread_->join();
    }
}

bool WebSocketClient::connect() {
    if (connected_.load()) {
        return true;
    }
    
    running_ = true;
    reconnecting_ = true;
    
    // Start IO thread if not already running
    if (!io_thread_) {
        io_thread_ = std::make_unique<std::thread>(&WebSocketClient::io_thread_loop, this);
    }
    
    // Connect on IO thread
    io_context_->post([this]() {
        websocket_ = std::make_unique<WebSocketImpl>(*io_context_, *ssl_context_);
        
        if (websocket_->connect(host_, port_)) {
            connected_.store(true);
            reconnecting_.store(false);
            connect_time_ = std::chrono::steady_clock::now();
            
            if (connection_callback_) {
                connection_callback_(true);
            }
            
            // Start reading messages
            websocket_->start_reading();
            
            // Send subscriptions
            send_subscription_message();
            
        } else {
            connected_.store(false);
            
            if (connection_callback_) {
                connection_callback_(false);
            }
            
            // Start reconnection timer
            start_reconnect_timer();
        }
    });
    
    // Wait for connection attempt
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    return connected_.load();
}

void WebSocketClient::disconnect() {
    running_ = false;
    connected_.store(false);
    
    if (websocket_) {
        io_context_->post([this]() {
            websocket_->disconnect();
        });
    }
}

bool WebSocketClient::subscribe_trades(const std::vector<std::string>& symbols) {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    
    try {
        nlohmann::json subscribe_msg = {
            {"op", "subscribe"},
            {"args", nlohmann::json::array()}
        };
        
        for (const auto& symbol : symbols) {
            subscribe_msg["args"].push_back("publicTrade." + symbol);
        }
        
        std::lock_guard<std::mutex> stat_lock(stats_mutex_);
        subscribed_symbols_ = symbols;
        
        if (connected_.load()) {
            send_message(subscribe_msg.dump());
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

bool WebSocketClient::subscribe_orderbook(const std::vector<std::string>& symbols, int depth) {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    
    try {
        std::string channel = "orderbook." + std::to_string(depth);
        nlohmann::json subscribe_msg = {
            {"op", "subscribe"},
            {"args", nlohmann::json::array()}
        };
        
        for (const auto& symbol : symbols) {
            subscribe_msg["args"].push_back(channel + "." + symbol);
        }
        
        subscribed_channels_.push_back(channel);
        
        if (connected_.load()) {
            send_message(subscribe_msg.dump());
        }
        
        return true;
    } catch (...) {
        return false;
    }
}

void WebSocketClient::unsubscribe_all() {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    
    try {
        nlohmann::json unsubscribe_msg = {
            {"op", "unsubscribe"},
            {"args", nlohmann::json::array()}
        };
        
        // Unsubscribe from all current subscriptions
        for (const auto& symbol : subscribed_symbols_) {
            unsubscribe_msg["args"].push_back("publicTrade." + symbol);
        }
        
        for (const auto& channel : subscribed_channels_) {
            unsubscribe_msg["args"].push_back(channel);
        }
        
        if (connected_.load()) {
            send_message(unsubscribe_msg.dump());
        }
        
        subscribed_symbols_.clear();
        subscribed_channels_.clear();
    } catch (...) {}
}

WebSocketClient::Statistics WebSocketClient::get_statistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    
    Statistics stats = stats_;
    
    if (connected_.load()) {
        auto now = std::chrono::steady_clock::now();
        stats.connection_uptime_ms += 
            std::chrono::duration_cast<std::chrono::milliseconds>(now - connect_time_).count();
    }
    
    return stats;
}

void WebSocketClient::reset_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_ = {0, 0, 0, 0, 1.0};
    connect_time_ = std::chrono::steady_clock::now();
}

void WebSocketClient::io_thread_loop() {
    while (running_.load()) {
        try {
            io_context_->run_for(std::chrono::milliseconds(100));
        } catch (const std::exception& e) {
            if (error_callback_) {
                error_callback_(std::string("IO thread error: ") + e.what());
            }
        }
    }
}

void WebSocketClient::on_message(const std::string& data) {
    try {
        WebSocketMessage msg = parse_message(data);
        
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.messages_received++;
        stats_.data_quality_score = std::max(0.0, 
            std::min(1.0, stats_.data_quality_score * 0.99 + 0.01)); // Decay
        
        if (message_callback_) {
            message_callback_(msg);
        }
        
        update_statistics();
    } catch (const std::exception& e) {
        if (error_callback_) {
            error_callback_(std::string("Message parsing error: ") + e.what());
        }
    }
}

WebSocketMessage WebSocketClient::parse_message(const std::string& data) {
    WebSocketMessage msg;
    
    try {
        auto json_data = nlohmann::json::parse(data);
        
        if (json_data.contains("topic") && json_data.contains("data")) {
            std::string topic = json_data["topic"];
            
            // Parse topic: "publicTrade.BTCUSDT"
            auto dot_pos = topic.find('.');
            if (dot_pos != std::string::npos) {
                msg.channel = topic.substr(0, dot_pos);
                msg.symbol = topic.substr(dot_pos + 1);
            }
            
            auto data_array = json_data["data"];
            if (data_array.is_array() && !data_array.empty()) {
                auto trade = data_array[0];
                
                msg.price = trade["price"];
                msg.volume = trade["size"];
                msg.side = trade["side"];
                msg.timestamp = trade["time"];
            }
        }
        
        msg.raw_data = data;
    } catch (...) {
        // Return empty message on parse error
    }
    
    return msg;
}

void WebSocketClient::send_subscription_message() {
    std::lock_guard<std::mutex> lock(subscription_mutex_);
    
    if (!subscribed_symbols_.empty()) {
        subscribe_trades(subscribed_symbols_);
    }
}

void WebSocketClient::send_message(const std::string& message) {
    if (websocket_ && connected_.load()) {
        io_context_->post([this, message]() {
            websocket_->send_message(message);
        });
    }
}

void WebSocketClient::start_reconnect_timer() {
    // Schedule reconnection attempt after delay
    io_context_->post([this]() {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        reconnect();
    });
}

void WebSocketClient::reconnect() {
    if (reconnecting_.load()) {
        return;
    }
    
    reconnecting_.store(true);
    
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.reconnection_count++;
    
    disconnect();
    
    // Attempt reconnection
    if (connect()) {
        if (connection_callback_) {
            connection_callback_(true);
        }
    } else {
        start_reconnect_timer();
    }
}

void WebSocketClient::update_statistics() {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    stats_.messages_processed = stats_.messages_received;
}

std::unique_ptr<WebSocketClient> create_websocket_client(const std::string& host,
                                                    const std::string& port,
                                                    bool use_ssl) {
    return std::make_unique<WebSocketClient>(host, port, use_ssl);
}

} // namespace mathcore
