# MotorLens frontend

Phase 4 is a Next.js interface that can be deployed to Vercel.

## Features

- Drag-and-drop or file-picker upload
- JPG, JPEG, PNG, and WEBP validation with a 10 MB browser-side limit
- Uploaded-image preview
- FastAPI loading and error states
- Primary make/model/generation result
- Five ranked candidates with confidence percentages
- Explicit uncertain/reference-supported state
- Transformer attention visualization
- Candidate confirmation without retaining the image
- Optional, explicit image contribution after confirmation
- Responsive keyboard, touch, tablet, and desktop layouts
- Privacy notice that dismisses on any screen click, its close button, or Escape

Dismissal of the privacy notice is not consent to image storage. The application only requests persistent storage after the user presses `Contribute this image`.

## Windows PowerShell setup

Run FastAPI from the repository root in the first PowerShell window:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

```powershell
.\.venv\Scripts\Activate.ps1
```

```powershell
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Open a second PowerShell window and move to the frontend:

```powershell
cd "path\to\AI_Car\frontend"
```

Install packages if they are not installed:

```powershell
npm install
```

Start the frontend:

```powershell
npm run dev
```

Open `http://localhost:3000`.

## Environment

The default backend URL is `http://127.0.0.1:8000`. To change it, copy `.env.example` to `.env.local` and set:

```text
NEXT_PUBLIC_API_URL=https://your-backend.example
```

For Vercel, set the same variable in the project settings and use `frontend` as the Root Directory.

## Testing and build

```powershell
npm test
```

The test command creates a production build and verifies the server-rendered product shell and explicit consent copy.

## Expected result

Uploading a supported vehicle image and selecting `Analyze vehicle` should show five candidates, the uncertainty message, and an attention visualization. Choosing a candidate and confirming it records only label feedback. `Contribute this image` resubmits the exact image with explicit consent so the backend can sanitize and queue it for review.

## Common errors

- `The AI service is unavailable`: start FastAPI on port 8000.
- Browser CORS error: confirm the frontend is running on `localhost:3000` or `127.0.0.1:3000`.
- Analysis takes a while: CPU model inference can take several seconds, especially on the first request.
- Feedback says the prediction expired: analyze the image again; prediction metadata expires after 30 minutes.
- Contribution says the image does not match: use the same original file that produced the prediction.
