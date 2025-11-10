#ifndef WEBSOCKET_CLIENT_H
#define WEBSOCKET_CLIENT_H

#include <string>
#include <functional>
#include <vector>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>

// Forward declarations for Boost.Asio
namespace boost {
namespace asio {
class io_context;
class ssl::context;
}
namespace system {
class error_code;
}
}

namespace mathcore {

struct WebSocketMessage {
    std::string symbol;
    std::string channel;
    double price;
    double volume;
    std::string side;
    std::int64_t timestamp;
    std::string raw_data;
};

class WebSocketClient {
public:
    WebSocketClient(const std::string& host = "stream.bybit.com",
                  const std::string& port = "443",
                  bool use_ssl = true);
    
    ~WebSocketClient();
    
    // Connection management
    bool connect();
    void disconnect();
    bool is_connected() const { return connected_.load(); }
    
    // Subscription management
    bool subscribe_trades(const std::vector<std::string>& symbols);
    bool subscribe_orderbook(const std::vector<std::string>& symbols, int depth = 25);
    bool subscribe_tickers(const std::vector<std::string>& symbols);
    void unsubscribe_all();
    
    // Callback registration
    using MessageCallback = std::function<void(const WebSocketMessage&)>;
    using ErrorCallback = std::function<void(const std::string&)>;
    using ConnectionCallback = std::function<void(bool)>;
    
    void set_message_callback(MessageCallback callback) { message_callback_ = callback; }
    void set_error_callback(ErrorCallback callback) { error_callback_ = callback; }
    void set_connection_callback(ConnectionCallback callback) { connection_callback_ = callback; }
    
    // Statistics
    struct Statistics {
        std::uint64_t messages_received;
        std::uint64_t messages_processed;
        std::uint64_t connection_uptime_ms;
        std::uint64_t reconnection_count;
        double data_quality_score;
    };
    
    Statistics get_statistics() const;
    void reset_statistics();

private:
    // Boost.Asio components
    std::unique_ptr<boost::asio::io_context> io_context_;
    std::unique_ptr<boost::asio::ssl::context> ssl_context_;
    std::unique_ptr<std::thread> io_thread_;
    std::atomic<bool> running_;
    
    // Connection state
    std::atomic<bool> connected_;
    std::atomic<bool> reconnecting_;
    std::string host_;
    std::string port_;
    bool use_ssl_;
    
    // WebSocket implementation
    class WebSocketImpl;
    std::unique_ptr<WebSocketImpl> websocket_;
    
    // Callbacks
    MessageCallback message_callback_;
    ErrorCallback error_callback_;
    ConnectionCallback connection_callback_;
    
    // Subscription management
    std::vector<std::string> subscribed_symbols_;
    std::vector<std::string> subscribed_channels_;
    mutable std::mutex subscription_mutex_;
    
    // Statistics
    mutable std::mutex stats_mutex_;
    Statistics stats_;
    std::chrono::steady_clock::time_point connect_time_;
    
    // Internal methods
    void io_thread_loop();
    void on_connect();
    void on_disconnect();
    void on_message(const std::string& data);
    void on_error(const std::string& error);
    void start_reconnect_timer();
    void reconnect();
    void send_subscription_message();
    void send_message(const std::string& message);
    WebSocketMessage parse_message(const std::string& data);
    void update_statistics();
};

// Factory function for Python integration
std::unique_ptr<WebSocketClient> create_websocket_client(const std::string& host,
                                                    const std::string& port,
                                                    bool use_ssl);

} // namespace mathcore

#endif // WEBSOCKET_CLIENT_H
