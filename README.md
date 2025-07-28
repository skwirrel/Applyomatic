# Auto‑apply (LLM)

Automates drafting a tailored covering letter and CV for a given role using OpenAI's Responses API with structured outputs. It keeps personal data in local config files and never prints secrets.

## Quick start

```bash
npm i
cp .env.example .env            # add your API key
cp config/cv.base.json.example config/cv.base.json
cp config/job.config.json.example config/job.config.json
cp config/covering-letter-notes.example.md config/covering-letter-notes.md
# edit the three config files with your details (keep them private)

npm run once                    # run a single attempt
npm run daemon                  # run continuously with waits between attempts
Outputs are printed to the console; redirect to files if you wish:

npm run once > output_$(date +%F).txt
