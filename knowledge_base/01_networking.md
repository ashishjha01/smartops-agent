# SmartOps Knowledge Base — Networking & Connectivity

## DNS Resolution Failures

If applications cannot resolve hostnames:

1. Verify DNS servers: `cat /etc/resolv.conf` (Linux) or `ipconfig /all` (Windows).
2. Test resolution: `nslookup api.example.com` or `dig api.example.com`.
3. Flush local DNS cache:
   - Linux systemd: `sudo systemd-resolve --flush-caches`
   - Windows: `ipconfig /flushdns`
4. Confirm the hostname exists in the corporate DNS zone or `/etc/hosts`.

Common root causes: stale cache, wrong search domain, VPN DNS split-tunnel misconfiguration.

## TCP Connection Timeouts

Symptoms: clients hang for 30–120s then fail with `Connection timed out`.

Checklist:
- Confirm the remote port is open (`telnet host 443` or `nc -vz host 443`).
- Check security groups / firewall rules allow egress and ingress.
- Verify the service is listening: `ss -tlnp | grep <port>`.
- Inspect MTU issues on VPN paths (try lowering MTU to 1400).

## HTTP 502 / 504 Gateway Errors

- **502 Bad Gateway**: upstream process crashed or refused the connection. Restart the app pool / container and check upstream health.
- **504 Gateway Timeout**: upstream took longer than the proxy timeout. Increase `proxy_read_timeout` (Nginx) or investigate slow DB queries.

## Recommended Tooling

Use `check_server_status(hostname)` when users report outages for a specific host. Combine with RAG docs before escalating.
