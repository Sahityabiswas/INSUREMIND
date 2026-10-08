# Local LLM Setup on Windows

`ollama is not recognized` means the executable is not installed or its installation folder is absent
from the current PowerShell session's PATH. Install the official package before using `ollama pull`.

This workspace uses D: for Ollama and model storage because C: has insufficient free space.

Expected installation and storage paths:

- `.runtime/ollama/ollama.exe`: installed Ollama CLI.
- `.runtime/models/`: downloaded models.
- `.runtime/tmp/`: temporary storage when the Python launcher starts the server.
- `.runtime/ollama-server.log`: server log when the Python launcher starts the server.

Once installed, open a new PowerShell window so it receives the updated PATH and model-storage setting.
If the current window still cannot find the command, use the executable directly:

```powershell
Set-Location 'D:\insurence seller\insurance_sales_agent'
$env:OLLAMA_MODELS = 'D:\insurence seller\insurance_sales_agent\.runtime\models'
& '.\.runtime\ollama\ollama.exe' --version
& '.\.runtime\ollama\ollama.exe' pull llama3.2:3b
& '..\term_project\Scripts\python.exe' run_llm.py --age 35 --budget mid --debug
```

`run_llm.py` starts a hidden local server if none is reachable, verifies that the requested model exists,
and runs the existing insurance chat. It allows up to 120 seconds for hardware detection at startup and
120 seconds for slower CPU generation. The server stays
available after the chat closes. Change the buyer age and budget as needed; type `/quit` to exit.

The debug field `generation_source=ollama` confirms an accepted model response. `llm_error` records
connection/timeout errors; `llm_rejected` indicates a rejected strategy or response. Those cases use templates.
Explicit rejection responses always use the deterministic closing template.

Sources: [Ollama Windows installation](https://docs.ollama.com/windows),
[official Windows download](https://ollama.com/download/windows),
[model library](https://ollama.com/library/llama3.2:3b).
