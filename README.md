# Slovak universal language model

A project for Natural Language Processing course.


## Requirements

It is assumed you are running a reasonably modern \*nix-like userspace (modern Linux distro, WSL, …)

You will need the following:

  - `curl`
  - `make`
  - `python3`


## Instructions

These instructions will detail how to run these tools to prepare the dataset and train the model.

### Prepare wikipedia

Run `make dataset/extracted-${LANG}wiki`, where `$LANG` is your language (by default `sk`). Please
note, that this may take a long time and cause a significant amount of network traffic.
