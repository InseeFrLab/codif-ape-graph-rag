#!/bin/bash

uv sync
uv run pre-commit install

export NEO4J_API_KEY=***
