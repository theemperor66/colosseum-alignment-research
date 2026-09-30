# Security

This repository contains research software, not a supported operational flight-control product. Its simulator connection can issue movement and reset commands. Keep simulator RPC private, use a single authorised control owner, and do not expose the unauthenticated RPC endpoint to the public internet.

Use only environments and records you are authorised to access. Do not commit credentials, keys, private network details, raw operational transcripts or restricted simulator assets. Software fixtures must be labelled synthetic. Authentic research examples may be published only with authorisation, a content review and explicit provenance; their presence does not authorise uploading unrelated raw records.

## Reporting a security concern

Do not post credentials, usable access details or sensitive records in a public issue. If GitHub offers **Report a vulnerability** on this repository's Security page, use that private reporting route. If it is unavailable, open a minimal issue requesting a private reporting channel without including exploit details or sensitive data. No private channel or response-time commitment is assumed by this policy.

A report should identify the affected release, the component, the security impact and a minimal reproduction that avoids third-party systems. General scientific validity concerns and non-sensitive software defects can be reported through ordinary issues.
