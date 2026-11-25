---
id: howto-custom-domain
title: Custom domains
---
Add a CNAME record pointing your domain to domains.northwind.example, then press Verify in Settings, then Domains.
DNS changes can take up to 30 minutes. After verification the TLS certificate is issued automatically within 15 minutes.
If verification keeps failing, check that the CNAME is on the exact hostname and has no conflicting A record. Guide: https://docs.northwind.example/custom-domains
