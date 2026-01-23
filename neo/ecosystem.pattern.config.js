module.exports = {
  apps: [{
    name: 'neo-pattern-bot',
    script: '/home/jbot/trading_ai/neo/pattern_bot.py',
    interpreter: 'python3',
    cwd: '/home/jbot/trading_ai/neo',
    instances: 1,
    autorestart: true,
    watch: false,
    max_memory_restart: '500M',
    env: {
      NODE_ENV: 'production',
      PYTHONUNBUFFERED: '1'
    },
    // Daily restart at midnight UTC
    cron_restart: '0 0 * * *',
    // Logging
    error_file: '/home/jbot/trading_ai/logs/pattern_bot_error.log',
    out_file: '/home/jbot/trading_ai/logs/pattern_bot.log',
    log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
    // Restart settings
    restart_delay: 5000,
    exp_backoff_restart_delay: 100
  }]
};
