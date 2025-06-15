[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTE3IDE2VjdsLTYgNU0yIDlWOGwxLTFoMWw0IDMgOC04aDFsNCAyIDEgMXYxNGwtMSAxLTQgMmgtMWwtOC04LTQgM0gzbC0xLTF2LTFsMy0zIi8+PC9zdmc+)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/superlinear-ai/wtpsplit-lite) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new/superlinear-ai/wtpsplit-lite)

# ✂️ wtpsplit-lite

🪓 [wtpsplit](https://github.com/segment-any-text/wtpsplit) is a Python package that offers training, inference, and evaluation of state-of-the-art Segment any Text (SaT) models for partitioning text into sentences.

✂️ [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite) is a lightweight version of wtsplit that only retains accelerated ONNX inference of SaT models with minimal dependencies:

1. [huggingface-hub](https://github.com/huggingface/huggingface_hub) to download the model
2. [numpy](https://github.com/numpy/numpy) to process the model in- and output
3. [onnxruntime](https://github.com/microsoft/onnxruntime) to run the model
4. [tokenizers](https://github.com/huggingface/tokenizers) to tokenize the text for the model

## Installing

To install this package, run:

```sh
pip install wtpsplit-lite
```

## Using

> [!TIP]
> For a complete list of Segment any Text (SaT) models and all `SaT.split` keyword arguments, see the [wtsplit README](https://github.com/segment-any-text/wtpsplit).

Example usage:

```python
from wtpsplit_lite import SaT

text = """
It is known that Maxwell’s electrodynamics—as usually understood at the
present time—when applied to moving bodies, leads to asymmetries which do
not appear to be inherent in the phenomena. Take, for example, the recipro-
cal electrodynamic action of a magnet and a conductor.
"""

# Fast (~150ms/page), good quality:
sat = SaT("sat-3l-sm")
sentences = sat.split(text, stride=128, block_size=256)

# Slow, highest quality:
sat = SaT("sat-12l-sm")
sentences = sat.split(text)
```

This package also [contributes a new 'hat' weighting scheme to wtpsplit](https://github.com/segment-any-text/wtpsplit/pull/147) that improves output quality when using large strides. To enable it, set `weighting="hat"` as follows:

```python
# Fast (~150ms/page), better quality:
sat = SaT("sat-3l-sm")
sentences = sat.split(text, stride=128, block_size=256, weighting="hat")
```

> [!NOTE]
> In wtpsplit, the SaT implementation treats newlines as sentence boundaries by default. However, this leads to poor results on text extracted from PDF such as in the example above. In wtpsplit-lite, newlines are therefore treated as whitepace by default. You can choose which behavior you prefer with the `treat_newline_as_space` boolean keyword argument of the `SaT.split` method.

## Contributing

<details>
<summary>Prerequisites</summary>

1. [Generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and [add the SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
1. Configure SSH to automatically load your SSH keys:

    ```sh
    cat << EOF >> ~/.ssh/config
    
    Host *
      AddKeysToAgent yes
      IgnoreUnknown UseKeychain
      UseKeychain yes
      ForwardAgent yes
    EOF
    ```

1. [Install Docker Desktop](https://www.docker.com/get-started).
1. [Install VS Code](https://code.visualstudio.com/) and [VS Code's Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Alternatively, install [PyCharm](https://www.jetbrains.com/pycharm/download/).
1. _Optional:_ install a [Nerd Font](https://www.nerdfonts.com/font-downloads) such as [FiraCode Nerd Font](https://github.com/ryanoasis/nerd-fonts/tree/master/patched-fonts/FiraCode) and [configure VS Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions) or [PyCharm](https://github.com/tonsky/FiraCode/wiki/Intellij-products-instructions) to use it.

</details>

<details open>
<summary>Development environments</summary>

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on [Open in GitHub Codespaces](https://github.com/codespaces/new/superlinear-ai/wtpsplit-lite) to start developing in your browser.
1. ⭐️ _VS Code Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/superlinear-ai/wtpsplit-lite) to clone this repository in a container volume and create a Dev Container with VS Code.
1. ⭐️ _uv_: clone this repository and run the following from root of the repository:

    ```sh
    # Create and install a virtual environment
    uv sync --python 3.10 --all-extras

    # Activate the virtual environment
    source .venv/bin/activate

    # Install the pre-commit hooks
    pre-commit install --install-hooks
    ```

1. _VS Code Dev Container_: clone this repository, open it with VS Code, and run <kbd>Ctrl/⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> → _Dev Containers: Reopen in Container_.
1. _PyCharm Dev Container_: clone this repository, open it with PyCharm, [create a Dev Container with Mount Sources](https://www.jetbrains.com/help/pycharm/start-dev-container-inside-ide.html), and [configure an existing Python interpreter](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#widget) at `/opt/venv/bin/python`.

</details>

<details open>
<summary>Developing</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `uv add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `uv.lock`. Add `--dev` to install a development dependency.
- Run `uv sync --upgrade` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`. Add `--only-dev` to upgrade the development dependencies only.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag. Then push the changes and the git tag with `git push origin main --tags`.

</details>
