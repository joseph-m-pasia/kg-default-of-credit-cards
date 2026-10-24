# Testing Guide

## 1. Overview

This document explains how to start, test, and troubleshoot the Credit Card Default Prediction application.

The project includes a machine learning model, a FastAPI backend, automated tests, Docker containerization, and a React frontend.

The testing process covers:

- Backend health and API functionality
- Machine learning prediction requests
- Frontend functionality
- Frontend-to-backend integration
- Automated testing and code quality
- Docker container execution
- Pre-commit verification

## 2. Prerequisites

Install the following tools before running the application locally:

- Python and pip
- Node.js and npm, if testing the React frontend
- Git
- Docker Desktop, if testing the containerized application
- Visual Studio Code or another code editor

Clone the repository if you have not already done so:

```bash
git clone https://github.com/joseph-m-pasia/kg-default-of-credit-cards.git
cd kg-default-of-credit-cards
```

The commands in this guide assume that the repository root contains `app/`, `tests/`, `pyproject.toml`, and `Dockerfile`.

The frontend instructions assume a React application in a `frontend/` directory. Adjust the paths if your actual project structure differs.

## 3. Start the Backend

### 3.1 Open a terminal

Open the repository in Visual Studio Code.

Open a terminal using:

`Terminal → New Terminal`

Make sure the terminal is in the project root.

### 3.2 Create a Python virtual environment

If you do not already have a virtual environment, create one:

```powershell
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script execution, use the activation method appropriate to your environment or activate the environment through VS Code.

### 3.3 Install project dependencies

Install the project and its dependencies:

```powershell
python -m pip install --upgrade pip
pip install -e .
```

If your project uses a `requirements.txt` file instead, follow the installation instructions in that file.

If pytest or Flake8 is not included in the installed dependencies, install the appropriate development dependencies according to your project configuration.

### 3.4 Start FastAPI

Start the backend from the project root:

```powershell
uvicorn app.main:app --reload
```

If the application starts successfully, Uvicorn should display a message similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

Keep this terminal running while testing the API.

If startup fails, inspect the terminal output for missing dependencies, incorrect module paths, or missing model artifacts.

## 4. Test the Backend

### 4.1 Test the health endpoint

Open the following URL in your browser:

http://127.0.0.1:8000/health

Alternatively, use PowerShell:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Expected result:

- The endpoint responds successfully.
- The response matches the health response implemented by the application.

A successful health check confirms that the endpoint is reachable. It does not necessarily prove that the prediction model is loaded correctly.

### 4.2 Open the interactive API documentation

Open:

http://127.0.0.1:8000/docs

FastAPI provides interactive API documentation through Swagger UI.

To test a prediction:

1. Locate the `POST /predict` endpoint.
2. Expand the endpoint.
3. Select **Try it out**.
4. Enter a valid JSON request.
5. Select **Execute**.
6. Inspect the HTTP status code and response body.

### 4.3 Test a prediction request

The following example is based on the input fields documented in the project README.

Use it only if the request schema in your running API accepts these fields.

```json
{
  "LIMIT_BAL": 20000,
  "SEX": 2,
  "EDUCATION": 2,
  "MARRIAGE": 1,
  "AGE": 24,
  "PAY_0": 2,
  "PAY_2": 2,
  "PAY_3": -1,
  "PAY_4": -1,
  "PAY_5": -2,
  "PAY_6": -2,
  "BILL_AMT1": 3913,
  "BILL_AMT2": 3102,
  "BILL_AMT3": 689,
  "BILL_AMT4": 0,
  "BILL_AMT5": 0,
  "BILL_AMT6": 0,
  "PAY_AMT1": 0,
  "PAY_AMT2": 689,
  "PAY_AMT3": 0,
  "PAY_AMT4": 0,
  "PAY_AMT5": 0,
  "PAY_AMT6": 0
}
```

Verify the following:

- The request is accepted by the API.
- The response matches the actual response schema.
- The prediction value is valid.
- Any returned probability is between 0 and 1.
- No unexpected exception appears in the backend terminal.

The prediction value depends on the trained model and input. Do not assume a particular result without executing the request.

### 4.4 Test invalid inputs

Submit requests with:

- Missing required fields
- Incorrect data types
- Invalid categorical values
- Values outside the ranges supported by the application

Expected result:

- Invalid requests are handled appropriately.
- Validation errors are returned where applicable.
- Invalid inputs do not cause unhandled server errors.

FastAPI commonly returns HTTP `422 Unprocessable Entity` when request validation fails.

The exact expected behavior depends on the validation rules implemented in the application.

## 5. Test the Frontend

This section applies if the React frontend is available in the repository.

### 5.1 Open a second terminal

Do not stop the backend terminal.

Open a second terminal in VS Code and navigate to the frontend directory:

```powershell
cd frontend
```

If the frontend is stored elsewhere, use its actual directory.

### 5.2 Install frontend dependencies

If you have not already installed the dependencies:

```powershell
npm install
```

This installs the packages declared in `package.json`.

### 5.3 Start the frontend development server

Run:

```powershell
npm run dev
```

Vite will display the local development URL. It is commonly:

http://localhost:5173

Open the URL shown in your terminal.

### 5.4 Check the user interface

Manually verify the following:

- The application loads without a blank screen.
- All expected input fields are visible.
- Field labels and input types are correct.
- Valid values can be entered.
- Required fields are validated.
- Invalid inputs display suitable messages.
- The Predict button responds to user interaction.
- Loading and error states are handled appropriately, if implemented.
- The prediction result is displayed correctly after a successful request.

### 5.5 Inspect browser errors

In Chrome:

1. Open the frontend.
2. Press `F12`.
3. Select **Console**.
4. Check for JavaScript errors.
5. Select **Network**.
6. Filter by **Fetch/XHR**.
7. Submit a prediction and inspect the API request.

Check the request URL, HTTP status code, request payload, and response body.

A frontend can render correctly even when its API integration is failing, so these checks are important.

## 6. Test Frontend-to-Backend Integration

Integration testing verifies that the React frontend communicates correctly with the FastAPI backend.

### 6.1 Start both applications

Keep both terminal windows running.

| Component | Typical local address |
|---|---|
| React frontend | `http://localhost:5173` |
| FastAPI backend | `http://127.0.0.1:8000` |
| FastAPI documentation | `http://127.0.0.1:8000/docs` |

Use the addresses displayed by your applications if they differ.

### 6.2 Submit a prediction through the frontend

1. Open the React application.
2. Enter valid customer information.
3. Click Predict.
4. Inspect the request in Chrome DevTools.
5. Verify that the backend receives the request.
6. Inspect the response returned by FastAPI.
7. Confirm that the frontend displays the response correctly.

A successful end-to-end test verifies the complete flow:

Customer input → React frontend → FastAPI endpoint → ML model → API response → displayed prediction.

### 6.3 Check error handling

Where applicable, test these situations:

- Backend is unavailable.
- The request contains invalid input.
- The API returns a validation error.
- The API returns a server error.
- The request takes longer than expected.

The frontend should provide understandable feedback rather than silently failing.

### 6.4 Check CORS configuration

If the browser reports a Cross-Origin Resource Sharing (CORS) error, verify the CORS middleware in the FastAPI application.

The backend should allow requests from the actual frontend origin used during development.

For example, if the frontend runs at `http://localhost:5173`, that origin may need to be included in the backend's allowed origins.

Do not disable CORS security controls indiscriminately to resolve the error.

## 7. Run Automated Backend Tests

The project uses pytest for backend testing.

From the repository root, run:

```powershell
python -m pytest -v
```

Pytest normally discovers tests in files named `test_*.py` or `*_test.py`.

Check the output:

- `PASSED` indicates that a test passed.
- `FAILED` indicates that a test assertion failed.
- `ERROR` indicates that a test could not execute successfully.

If tests fail, read the traceback and investigate the underlying issue before proceeding.

### 7.1 Recommended backend test coverage

The backend test suite should cover:

- Health endpoint responses
- Valid prediction requests
- Invalid or missing input fields
- Response schema validation
- Prediction output validation
- Probability range validation, if probabilities are returned
- Model loading and error handling

Tests should verify actual application behavior rather than simply confirming that functions execute.

## 8. Run Code Quality Checks

The project uses Flake8 for Python code quality checks.

From the repository root, run:

```powershell
flake8 .
```

Review any reported style violations.

If the project has a Flake8 configuration, follow the rules defined in that configuration.

Resolve relevant issues before committing changes.

## 9. Test with Docker

Docker testing verifies that the backend can run in a container.

Make sure Docker Desktop is running before executing these commands.

### 9.1 Build the Docker image

From the directory containing the Dockerfile:

```powershell
docker build -t credit-default-api .
```

Wait for the build to finish successfully.

### 9.2 Run the container

```powershell
docker run --rm -p 8000:8000 credit-default-api
```

Keep the container running.

If another application is already using port 8000, stop that application or map the container to a different host port.

### 9.3 Test the containerized API

Open:

http://127.0.0.1:8000/health

Then open:

http://127.0.0.1:8000/docs

Test the prediction endpoint in Swagger UI.

Confirm that the container can load the required model artifacts and return predictions.

### 9.4 Stop the container

Press `Ctrl + C` in the terminal running the container.

The `--rm` option removes the stopped container automatically.

## 10. Continuous Integration

The repository includes a GitHub Actions workflow at:

`.github/workflows/ci.yml`

Review the workflow to confirm which checks are actually configured.

Depending on the workflow configuration, CI may perform tasks such as:

- Installing Python dependencies
- Running pytest
- Running Flake8
- Building a Docker image

To verify CI:

1. Commit and push your changes.
2. Open the repository on GitHub.
3. Select the **Actions** tab.
4. Open the relevant workflow run.
5. Review the status of each job and step.
6. Investigate failures before merging changes.

A successful CI run confirms that the configured checks passed. It does not automatically prove that the application was deployed or that a production environment works correctly.

## 11. Troubleshooting

| Symptom | Possible cause | Recommended check |
|---|---|---|
| `ModuleNotFoundError` | Missing dependency or incorrect environment | Activate the virtual environment and install dependencies |
| `uvicorn` command not found | Uvicorn is not installed in the active environment | Install the project dependencies |
| Health endpoint returns `404` | Health route is missing or uses another path | Inspect `app/main.py` and `/docs` |
| `Connection refused` | Backend is not running or the port is incorrect | Check the backend terminal and URL |
| HTTP `404` for predictions | Incorrect endpoint path | Check the frontend request URL and API routes |
| HTTP `422` | Request fields or types do not match the API schema | Compare the payload with `/docs` |
| HTTP `500` | Backend prediction or model-loading error | Inspect the backend traceback |
| `Failed to fetch` | Backend unavailable, incorrect URL, or browser/network issue | Inspect DevTools and the backend terminal |
| CORS error | Frontend origin is not allowed | Check FastAPI CORS configuration |
| Prediction is not displayed | Frontend response-handling problem | Inspect the Network response and frontend code |
| Docker build fails | Dependency, path, or Dockerfile issue | Inspect the build output |
| Model file not found | Artifact missing from the container or incorrect path | Check artifact paths and Docker image contents |
| Pytest fails | Test assertion, setup, dependency, or application issue | Read the traceback and reproduce the failure |

## 12. Pre-Commit Checklist

Before committing or pushing changes, verify the relevant items below.

### Backend

- [ ] Python environment is activated.
- [ ] Dependencies are installed.
- [ ] FastAPI starts successfully.
- [ ] Health endpoint responds successfully.
- [ ] Prediction endpoint accepts valid input.
- [ ] Invalid input is handled appropriately.
- [ ] Automated backend tests pass.
- [ ] Flake8 checks pass.

### Frontend

- [ ] Frontend dependencies are installed.
- [ ] Development server starts successfully.
- [ ] The page loads without unexpected errors.
- [ ] Form validation works.
- [ ] API requests are sent to the correct backend.
- [ ] Successful predictions are displayed correctly.
- [ ] Error states are handled appropriately.
- [ ] Automated frontend tests pass, if configured.

### Integration and deployment

- [ ] End-to-end prediction works locally.
- [ ] Browser console and Network tab have been checked.
- [ ] Docker image builds successfully, when applicable.
- [ ] Containerized API responds correctly, when applicable.
- [ ] GitHub Actions checks pass.

## 13. Useful Commands Reference

| Task | Command |
|---|---|
| Activate Python environment | `.\.venv\Scripts\Activate.ps1` |
| Install Python project | `pip install -e .` |
| Start FastAPI | `uvicorn app.main:app --reload` |
| Run backend tests | `python -m pytest -v` |
| Run Flake8 | `flake8 .` |
| Install frontend dependencies | `npm install` |
| Start frontend | `npm run dev` |
| List available npm scripts | `npm run` |
| Build Docker image | `docker build -t credit-default-api .` |
| Run Docker container | `docker run --rm -p 8000:8000 credit-default-api` |

The commands assume that the corresponding tools, dependencies, scripts, and configuration are present in the project.

## 14. Related Documentation

- [Project README](README.md)
- [GitHub repository](https://github.com/joseph-m-pasia/kg-default-of-credit-cards)
- [FastAPI documentation](https://fastapi.tiangolo.com/)
- [pytest documentation](https://docs.pytest.org/)
- [Docker documentation](https://docs.docker.com/)
- [Vite documentation](https://vite.dev/)

---

**Purpose:** This guide supports repeatable local testing and troubleshooting of the Credit Card Default Prediction project. Update it whenever the application structure, commands, test suites, or deployment process changes.
