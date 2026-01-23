// PM2 Ecosystem Configuration for Pre-Market Report
// Runs at 6:00 AM UTC Monday-Friday before London open

module.exports = {
  apps: [{
    name: "premarket-report",
    script: "/home/jbot/trading_ai/neo/premarket_report.py",
    interpreter: "python3",
    args: "--send",  // Send to Telegram
    instances: 1,
    autorestart: false,  // Don't restart - it's a one-shot cron job
    watch: false,
    
    // Run at 6:00 AM UTC, Monday-Friday
    cron_restart: "0 6 * * 1-5",
    
    env: {
      TELEGRAM_BOT_TOKEN: "8250652030:AAFd4x8NsTfdaB3O67lUnMhotT2XY61600s",
      ADMIN_CHAT_ID: "6776619257"
    },
    
    error_file: "/home/jbot/trading_ai/logs/premarket_error.log",
    out_file: "/home/jbot/trading_ai/logs/premarket.log",
    log_date_format: "YYYY-MM-DD HH:mm:ss Z",
    
    // PM2 Plus monitoring
    max_memory_restart: "200M",
    
    // Additional schedule notes
    // This sends the pre-market MM analysis before London session (08:00 UTC)
    // Gives 2 hours to review and plan trades
  }]
};
