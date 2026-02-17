find-tunes/
├── docker-compose.yml          <-- Best place: Absolute Root
├── Dockerfile                  <-- Where we install FFmpeg
├── pyproject.toml
├── .env                        <-- Where we store DATABASE_URL safely
├── frontend/
│        ├── index.html       # The Skeleton (Buttons, Divs)
│        ├── style.css        # The Skin (Animations, Colors)
│        └── app.js           # The Brain (WebSocket logic, Audio Recorder)
└── src/
    └── find_tunes/             
        ├── __init__.py
        ├── core/
        │   ├── config.py
        │   └── database.py
        ├── services/
        │   ├── audio_engine.py
        │   ├── ml_processor.py
        │   └── ingestion.py
        └── api/
            └── main.py         <-- FastAPI App

============================================================
📊 FIND-TUNES A/B SYSTEM EVALUATION REPORT
============================================================
MODE       | METRIC          | DSP (Shazam)    | ML (Fusion)
------------------------------------------------------------
CLEAN      | Top-1 Accuracy  |  49.4% (396/801)   |  48.2% (386/801)
           | Top-5 Accuracy  |  74.7% (598/801)   |  88.9% (712/801)
           | Offset (±2s)    |  95.2%           |  93.8%
------------------------------------------------------------
SOFT       | Top-1 Accuracy  |  43.6% (349/801)   |  40.6% (325/801)
           | Top-5 Accuracy  |  68.8% (551/801)   |  84.3% (675/801)
           | Offset (±2s)    |  92.8%           |  88.6%
------------------------------------------------------------
HARD       | Top-1 Accuracy  |  11.2% (90/801)   |   8.7% (70/801)
           | Top-5 Accuracy  |  22.6% (181/801)   |  43.4% (348/801)
           | Offset (±2s)    |  81.1%           |  80.0%
------------------------------------------------------------
ULTRA      | Top-1 Accuracy  |   1.0% (8/801)   |   1.6% (13/801)
           | Top-5 Accuracy  |   4.7% (38/801)   |  15.9% (127/801)
           | Offset (±2s)    |  62.5%           |  53.8%
------------------------------------------------------------

Noise Level,DSP Top-5 Accuracy,ML Fusion Top-5 Accuracy,ML Improvement (%)
CLEAN,74.7%,88.9%,+19.0%
SOFT,68.8%,84.3%,+22.5%
HARD,22.6%,43.4%,+92.0%
ULTRA,4.7%,15.9%,+238.3%

REMEMBER 
--need to build ffmpeg with the container using 
    RUN apt-get update && apt-get install -y \  ffmpeg \  libpq-dev \  gcc \    && rm -rf /var/lib/apt/lists/*


TO DO 
--for the continual learning part 
Phase 1: The State Migration (AWS RDS)

        You must do this first. Both your local laptop (running the FastAPI web app) and your EC2 (running the continual learning) must talk to the exact same database.

        Create an AWS RDS Instance: Provision a PostgreSQL database.

        Enable pgvector: You will need to execute CREATE EXTENSION vector; on the RDS instance.

        Security Groups: Whitelist your local IP address and your future EC2's IP address so they can connect to port 5432.

        Data Push: Use pg_dump and pg_restore (or your seed_DB.py script) to move your local findtunes data into the RDS instance.

Phase 2: Compute Provisioning (AWS EC2)

        Launch Instance: Select a machine capable of handling PyTorch. A g4dn.xlarge (with NVIDIA T4 GPU) is best if you want it done in minutes, or a beefy CPU instance like t3.large if you are willing to wait hours for the cron job to finish.

        Select AMI: Choose an Ubuntu Deep Learning AMI (Amazon Machine Image). This saves you the nightmare of manually installing NVIDIA drivers and CUDA toolkits.

        Configure Storage: Allocate at least 50GB of EBS Storage. Audio files, .pth checkpoints, and Docker images will eat up a standard 8GB drive instantly.

Phase 3: The Runtime Environment (EC2 Terminal)

        Once you SSH into the EC2, you must prepare the environment to mirror your local machine.

        Install uv: Run the curl command to install the Astral uv package manager.

        Install Git: Required to clone your repository.

        Audio Dependencies: Run sudo apt-get install ffmpeg libsndfile1 (Crucial for soundfile and torchaudio to process .wav files without crashing).

Phase 4: The CI/CD Bridge (GitHub Actions)

        Generate Runner Token: Go to your GitHub Repo -> Settings -> Actions -> Runners -> "New self-hosted runner".

        Install the Runner: Copy/paste the exact 5 commands GitHub provides into your EC2 terminal to download and configure the runner agent.

        Start the Service: Run sudo ./svc.sh install and sudo ./svc.sh start so the GitHub listener runs constantly in the background, even if you close your SSH session.

Phase 5: Secrets & Security

        GitHub Secrets: Go to Repo Settings -> Secrets and Variables -> Actions.

        Add Variables: You must add DATABASE_URL (pointing to your new AWS RDS endpoint) and DAGSHUB_TOKEN. If you don't do this, the YAML file we wrote today will fail on step 3