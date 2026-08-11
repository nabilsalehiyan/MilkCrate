# MilkCrate Backend — Hetzner Deployment Guide

## 1. Create Hetzner server
- Go to console.hetzner.cloud → New Project → Add Server
- Location: Ashburn, VA (US East) — closest to most US users
- Image: Ubuntu 24.04
- Type: CX22 (2 vCPU, 4GB RAM) — €4.55/mo
- Add your SSH key
- Create

## 2. Initial setup (SSH in as root)
```bash
apt update && apt upgrade -y
apt install -y python3-pip python3-venv postgresql nginx certbot python3-certbot-nginx

# Create app user
useradd -m -s /bin/bash milkcrate
mkdir -p /opt/milkcrate/models
chown -R milkcrate:milkcrate /opt/milkcrate
```

## 3. PostgreSQL setup
```bash
sudo -u postgres psql -c "CREATE USER milkcrate WITH PASSWORD 'CHANGE_ME';"
sudo -u postgres psql -c "CREATE DATABASE milkcrate OWNER milkcrate;"
```

## 4. Deploy app
```bash
su - milkcrate
git clone <your-repo> app   # or scp the files
cd app
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 5. Systemd service
Create /etc/systemd/system/milkcrate.service:
```ini
[Unit]
Description=MilkCrate Backend
After=network.target postgresql.service

[Service]
User=milkcrate
WorkingDirectory=/home/milkcrate/app
Environment=DATABASE_URL=postgresql://milkcrate:CHANGE_ME@localhost/milkcrate
Environment=MODEL_DIR=/opt/milkcrate/models
Environment=API_KEYS=beta-key-1,beta-key-2
ExecStart=/home/milkcrate/app/venv/bin/uvicorn main:app --host 127.0.0.1 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

```bash
systemctl enable --now milkcrate
```

## 6. Nginx + HTTPS
```bash
# /etc/nginx/sites-available/milkcrate
server {
    server_name api.yourdomain.com;
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
    }
}

ln -s /etc/nginx/sites-available/milkcrate /etc/nginx/sites-enabled/
certbot --nginx -d api.yourdomain.com
systemctl reload nginx
```

## 7. Weekly retraining cron
```bash
crontab -e -u milkcrate
# Add:
0 3 * * 0 cd /home/milkcrate/app && venv/bin/python retrain_server.py >> /var/log/milkcrate-retrain.log 2>&1
```

## 8. Upload base features
scp your enhanced_features.parquet to /opt/milkcrate/base_features.parquet
