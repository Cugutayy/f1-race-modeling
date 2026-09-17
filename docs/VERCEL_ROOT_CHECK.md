# Vercel root deploy check

The repository can be imported into Vercel at the repository root. Root `vercel.json`
installs and builds the Next.js app under `web/` and publishes `web/.next`.

This exists as a deployment guard because the Vercel project may be imported before a
Root Directory is configured. The preferred dashboard setting is still Root Directory
`web`, but the root configuration must also remain buildable.

Required server-side environment variables:

- `F1_BACKEND_URL`
- `F1_BACKEND_TOKEN`

The persistent OpenF1 capture/model/API worker is not hosted in Vercel.
