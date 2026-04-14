# Security Policy

## Scope

This repository is an educational and research platform for transformer diagnostics.
Production deployments must perform independent security review and infrastructure hardening.

## Reporting A Vulnerability

Please report vulnerabilities privately by email:

- [contact@aletheia-sovereign.com](mailto:contact@aletheia-sovereign.com)

Include:

- Affected file or component
- Reproduction steps
- Impact assessment
- Suggested remediation if available

Do not open a public issue for unpatched vulnerabilities.

## Threat Model Summary

### Assets

- User-provided text in the Self-Serve Audit Hub
- Diagnostic outputs and benchmark artifacts
- Notebook execution environments

### Trust Boundaries

- Browser local analysis path in [index.html](index.html)
- Optional external proxy endpoint: [geometric-brain-mcp onrender health-check](https://geometric-brain-mcp.onrender.com/v1/brain/health-check)
- External font delivery via Google Fonts

### Primary Risks

- Sensitive data accidentally sent through external proxy mode
- Client-side injection risk in dynamic UI rendering
- Dependency and notebook supply-chain risk in CI/runtime environments

### Current Mitigations

- External proxy mode is opt-in and disabled by default in the audit UI
- Optional client-side redaction is available before outbound proxy request
- CSP is defined in page metadata to restrict script, style, font, frame, and connect sources
- UI rendering uses safe DOM APIs for user-derived content paths
- Notebook CI validates structure and executes the starter notebook on pull requests and pushes

## Data Handling Notes

- Local spectral analysis runs in-browser and does not require outbound transmission.
- External text proxy mode sends request payloads to the endpoint above only after explicit user opt-in.
- Optional redaction can remove common sensitive patterns (email, phone, card-like number, token-like strings) before outbound request construction.

## Security Maintenance

- Keep dependencies current and review release notes for security advisories.
- Re-run tests and notebook checks for security-sensitive UI or workflow changes.
- Review CSP and outbound endpoint allowlists whenever new integrations are added.
