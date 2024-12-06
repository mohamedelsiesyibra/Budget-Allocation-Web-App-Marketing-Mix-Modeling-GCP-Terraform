# Budget Allocation Web App | Marketing Mix Modeling | GCP | Terraform

## Overview

This project demonstrates the end-to-end workflow of a Marketing Mix Modeling (MMM) pipeline using a lightweight MMM (LightweightMMM) library on Google Cloud Platform (GCP). It aims to empower marketing teams and analysts to better understand the effectiveness of various marketing channels and inform optimal budget allocations.

## Web App Demo

🚀 [Live Demo](https://mmm-app-mohamed-elsiesy.streamlit.app/) 

![Streamlit Web App](images/web-app.png)
*Budget Allocation Web Interface*

## Architecture

![System Architecture](images/architecture.png)
*System Architecture Diagram*

## Key Components

- **Data Warehouse (Google BigQuery):** Stores historical marketing spend and performance data
- **Marketing Mix Modeling (LightweightMMM):** Python-based Bayesian modeling library
- **Model Training & Hosting (Cloud Run):** Serverless container environment
- **Storage Layer (Google Cloud Storage):** Model and artifact storage
- **Visualization & Budget Allocation (Streamlit):** Interactive web interface
- **Infrastructure as Code (Terraform):** Automated deployment

## Features

- 🚀 Scalable & Serverless architecture
- 📊 Modular data flows with BigQuery
- ⚙️ Configurable model structure
- 📈 Interactive budget optimization
- 🔄 Automated infrastructure deployment


## Tech Stack

- **Google BigQuery:** Data warehousing and SQL transformations
- **LightweightMMM (Python):** Bayesian MMM modeling
- **Google Cloud Run:** Serverless container execution for both the model and front-end app
- **Google Cloud Storage:** Model artifact and data storage
- **Streamlit:** Interactive UI for budget allocation scenarios
- **Terraform:** Infrastructure as code for consistent and reliable deployments
