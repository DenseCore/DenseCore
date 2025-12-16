# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 2.x.x   | :white_check_mark: |
| < 2.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in DenseCore, please report it responsibly.

### How to Report

**Please do NOT open public issues for security vulnerabilities.**

Instead, send your report via email to: **security@densecore.io**

### What to Include

- Description of the vulnerability
- Steps to reproduce (if applicable)
- Potential impact assessment
- Any suggested fixes (optional)

### Response Timeline

| Action | Timeframe |
|--------|-----------|
| Acknowledgment | Within 48 hours |
| Initial assessment | Within 7 days |
| Status update | Every 14 days |
| Fix release | ASAP (based on severity) |

### Disclosure Policy

- We follow a **90-day disclosure policy** for non-critical vulnerabilities
- Critical vulnerabilities will be addressed immediately
- We will credit reporters (unless anonymity is requested) in release notes

### Scope

This security policy applies to:

- `core/` - C++ inference engine
- `python/` - Python SDK
- `server/` - Go REST server
- Docker images published on Docker Hub

### Out of Scope

- Vulnerabilities in dependencies (please report to upstream)
- Social engineering attacks
- Physical attacks
- Issues in experimental/unreleased features

## Security Best Practices

When deploying DenseCore in production:

1. **Enable Authentication**: Set `AUTH_ENABLED=true` and use strong API keys
2. **Use HTTPS**: Place behind a reverse proxy with TLS
3. **Network Isolation**: Run in private networks, expose only through ingress
4. **Resource Limits**: Set appropriate CPU/memory limits in K8s
5. **Regular Updates**: Keep DenseCore and dependencies up to date
