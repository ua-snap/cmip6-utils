# CMIP6 Transfer / downloading pipelines

This folder will contain pipelines transferring raw CMIP6 data to SNAP/UAF infrastructure. 

## Globus CLI

This codebase makes use of the Globus CLI for transferring CMIP6 data from ESGF nodes to SNAP's Arctic Climate Data Node.

Here are some steps on using the Globus CLI and to perform a transfer. Steps 1-3 need to be done before running any python scripts that make use of this CLI.

1. Activate the `cmip6-utils` conda environment which should have the Globus CLI already installed (but you can run e.g. `globus version` to check).
2. Login with `globus login`. Copy the link and paste into web browser to login and retrieve a key.
3. Grant consent to your instance of the CLI for our UAF endpoint. The UAF endpoint is 7235217a-be50-46ba-be31-70bffe2b5bf4. Grant consent like so:

```
globus session consent 'urn:globus:auth:scope:transfer.api.globus.org:all[*https://auth.globus.org/scopes/7235217a-be50-46ba-be31-70bffe2b5bf4/data_access]'
```

You will again have to copy a link into a browser and retrieve another key.

You can verify this worked by making sure `globus ls 7235217a-be50-46ba-be31-70bffe2b5bf` returns sensible information. 

4. Activate an endpoint you will be transferring from if it has not been done recently. Activations are temporary and will likely need to be done after some period of inactivity. Activate like so:

```
globus endpoint activate --web <endpoint id>
```

5. Execute a transfer command like so.

```
globus transfer 415a6320-e49c-11e5-9798-22000b9da45e 7235217a-be50-46ba-be31-70bffe2b5bf4 --label "CLI Batch" --batch /tmp/transferList_415a6320-e49c-11e5-9798-22000b9da45e_5206.txt
```

This transfers all files listed in `/tmp/transferList_415a6320-e49c-11e5-9798-22000b9da45e_5206.txt` (structured as `path/on/source path/on/target`) from endpoint 415a6320-e49c-11e5-9798-22000b9da45e (LLNL ESGF) to 415a6320-e49c-11e5-9798-22000b9da45e (our endpoint, ACDN).

6. log out of globus via `globus logout`

