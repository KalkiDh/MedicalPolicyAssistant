# Medical Policy Assistant

This project is a medical policy assistant that allows users to upload documents and ask questions about them.

## Running the project locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/KalkiDh/MedicalPolicyAssistant.git
   cd MedicalPolicyAssistant
   ```

2. **Create a `.env` file:**
   Create a `.env` file in the root of the project and add your GitHub token:
   ```
   GITHUB_TOKEN="your_github_token"
   ```

3. **Install the dependencies:**
   Make sure you have Python 3.10 or higher installed.
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   uvicorn app:app --reload
   ```
   The application will start on `http://127.0.0.1:8000`.

5. **Access the frontend:**
   Once the application is running, you can access the frontend at [https://medicalassistdron.netlify.app/](https://medicalassistdron.netlify.app/).

