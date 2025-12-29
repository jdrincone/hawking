# Galileo: Data Analysis and Report Generation System 🚀📊📝

Galileo is an exploration data analysis platform and report generation system designed to transform business inquiries into actionable insights through dynamic HTML reports. The system combines data analysis capabilities with automated report generation using YAML configuration files.

## Overview

Galileo provides a streamlined workflow for:
- Data exploration and analysis
- Visualization of insights
- Automated report generation
- Dynamic HTML report creation with embedded visualizations

## System Architecture

The project is organized into the following key components:

```
hawking/
├── automation/          # Specialized automation for specific projects
├── core/               # Core functionality
│   ├── build_report.py # Report generation engine
│   ├── report.py       # Report utilities and asset processing
│   ├── s3.py           # S3 management and AWS integration
│   ├── viz.py          # Visualization library (Plotly)
│   ├── templates/      # Jinja2 HTML templates
│   └── static/         # Static assets (logos, icons)
├── exploration/        # Jupyter notebooks for EDA and analysis
├── yamls/              # YAML configuration files for reports
├── raw/                # Raw data files and Excels
├── data/               # Processed data artifacts
├── images/             # Generated visualizations and plots
└── reports/            # Generated HTML reports (local output)
├── .env                # Environment variables (AWS/S3)
└── pyproject.toml      # Dependency management with uv
```

## How to use

### 1. Setup Environment
First, ensure you have your environment variables configured.

```bash
.env
```

### 2. Dependency Management
This project uses `uv` for fast and reliable dependency management. To install all requirements and setup the virtual environment:

```bash
uv sync
```

### 3. Generate a Report

To generate a report from a YAML configuration file located in the `yamls/` directory, run:

```bash
uv run python -m core.build_report --name your_report.yaml
```

**Example:**
```bash
uv run python -m core.build_report --name aliforte_production.yaml
```

The system will:
1. Load the configuration from `yamls/aliforte_production.yaml`.
2. Process the specified sections (images, tables, charts).
3. Render the HTML using the base template.
4. Save the result in the `reports/` directory.
5. Automatically open the report in your browser.
6. Upload the report to S3 (if `S3_BUCKET` is configured).

## Documentation

This project uses **Quarto** for comprehensive documentation, including business logic, data science workflows, and technical references.

### 0. Install Quarto
If you don't have Quarto installed yet, you can install it using Homebrew:

```bash
brew install --cask quarto
```

### 1. Preview Locally
To serves and preview the documentation on your local machine with live-reloading:

```bash
quarto preview docs
```

### 2. Deploy to S3
To render and upload the documentation to the configured S3 bucket (loads `S3_BUCKET` from `.env`):

```bash
./deploy_docs.sh
```
