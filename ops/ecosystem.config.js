module.exports = {
  apps: [
    {
      name: 'battle-ops-scheduler',
      script: 'scheduler.py',
      interpreter: 'python3',
      cwd: '/home/jbot/trading_ai/ops',
      env: {
        TELEGRAM_BOT_TOKEN: process.env.TELEGRAM_BOT_TOKEN,
        TELEGRAM_CHAT_ID: process.env.TELEGRAM_CHAT_ID,
      },
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      error_file: '/home/jbot/.pm2/logs/battle-ops-error.log',
      out_file: '/home/jbot/.pm2/logs/battle-ops-out.log',
    }
  ]
};
