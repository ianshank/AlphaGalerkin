# Security Policy

## Supported Versions

AlphaGalerkin is under active development and has not yet cut a tagged release.
Security fixes are applied to the latest commit on the repository's default branch.

| Version | Supported |
| ------- | --------- |
| Default branch (latest) | :white_check_mark: |
| older commits           | :x: |

## Reporting a Vulnerability

**Please do not open a public issue for security vulnerabilities.**

1. **Preferred:** Use GitHub's [private vulnerability reporting](https://github.com/ianshank/AlphaGalerkin/security/advisories/new)
   (the **Security** tab → *Report a vulnerability*). This keeps the report private
   until a fix is available.
2. **Fallback:** Email **ianshank@gmail.com** with the subject line
   `SECURITY: AlphaGalerkin`.

Please include:

- A description of the vulnerability and its impact.
- Steps to reproduce (a minimal proof of concept if possible).
- Affected module(s) / commit hash.

### What to expect

- **Acknowledgement** within 5 business days.
- A **remediation plan** or request for more information within 10 business days.
- Public disclosure (crediting the reporter, unless anonymity is requested) once
  a fix is merged to the default branch.

## Scope

This policy covers the code in this repository. Dependencies are pinned in
`pyproject.toml`; vulnerabilities in third-party packages should be reported to
their respective maintainers, though we welcome a heads-up so we can bump the pin.
