# Apps Script

This directory contains the Google Apps Script source code that runs in
Google's cloud alongside the Python experiment log tools.

The Python side (`logmaker_4_googledocs/docgen.py`) calls these functions via
the Apps Script API, using the `SCRIPT_ID` stored in
`logmaker_4_googledocs/config.ini`.

## Deploying / updating the script

1. Open [script.google.com](https://script.google.com) and select the
   existing project (or create a new one if starting fresh).
2. Replace the editor contents with the latest `Code.gs` from this directory.
3. **Deploy → Manage deployments → New version** to publish the changes.
4. The `SCRIPT_ID` in `config.ini` does **not** change between versions —
   it identifies the project, not a specific deployment.

## Linking a new project

If you need to set up a fresh Apps Script project:

1. Create a new project at [script.google.com](https://script.google.com).
2. Paste `Code.gs` into the editor and save.
3. Deploy as an API executable (**Deploy → New deployment → API executable**).
4. Copy the **Script ID** from the deployment settings.
5. Add it to `logmaker_4_googledocs/config.ini`:

```ini
[DEFAULT]
script = <your_script_id_here>
```
