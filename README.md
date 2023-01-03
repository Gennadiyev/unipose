![logo](docs/unipose.svg)

# Unipose

[![stable-docs](https://shields.io/badge/docs-stable-blue.svg)](https://gennadiyev.github.io/unipose/apidocs) [![style-black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.vercel.app/) [![python>=3.8](https://img.shields.io/badge/python->=3.8-green.svg)](https://www.python.org/downloads/) [![license](https://img.shields.io/badge/license-MIT-blue.svg)](https://mit-license.org/)

**Bridging the poses of humans and tetrapods with one general model.**

## Capabilities

- **Pose estimation**: Unipose can estimate the pose of any tetrapod with high accuracy.
- **Pose tracking**: Unipose can track the pose of any tetrapod in a video sequence. (TODO)
- **Animal Classification**: Unipose can classify the species of any tetrapod based on its pose. (TODO)

## Features

- **Generalized model**: Unipose is a generalized model that can be applied to any tetrapod species.
- **Code quality**: Unipose utilizes a modern python development toolchain using [poetry](https://python-poetry.org/) for packaging, [black](https://black.vercel.app/) for code style enforcement, [pytest](https://pytest.org/) for automated testing and [pydoctor](https://pydoctor.readthedocs.io/en/latest/) for documentation.

## Usage

A front-end server supporting API calls is planned for the future. For now, see [`scripts/vis_graph.py`](scripts/vis_graph.py) for an example of how to use the model.

## Pretrained Models

| Model | Specialty |
| --- | --- |
| [`model_run-5dd8_ep-60.pth`](https://drive.google.com/drive/folders/1eJ9RyLHcezrxE02uHBNmrI0Wxg4OPkfQ?usp=sharing) | Generalized model for all tetrapods (humans and animals) |

## Evaluation

(Coming soon, but you can refer to `scripts/test.py`.)

## Maintainers

- [Yikun Ji (Kunologist)](https://github.com/Gennadiyev)
- [Qi Fan (fanqiNO1)](https://github.com/fanqiNO1)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for more information.

