# Development Guide for a New Module

## Setup and Configuration

### 1. Update Dockerfile
- Download necessary files for build
- Point to downloaded files (runtime won't have internet access)
- Ensure all dependencies are properly installed

### 2. Update `main.py`
- Add your module's core logic
- Access input through environment variables:
  ```python
  input = os.environ.get("INPUT") or "default value"
  ```
- Implement error handling and logging

### 3. Update `requirements.txt`
- List all Python package dependencies
- Specify version numbers for consistency

### 4. Configure `config.env`
- Set your image name and tag:
  ```
  # Generic example
  IMAGE_NAME=your-module-name
  IMAGE_TAG=v1.0.0

  # Specific example
  IMAGE_NAME=test
  IMAGE_TAG=synth-data
  ```

### 5. Leave `start.sh` and `build.sh` unchanged
- These scripts handle standard operations

## Testing and Deployment

### Local Docker Testing

1. Build the Docker image:
   ```bash
   # Generic example
   docker build --progress=plain -f Dockerfile -t your-module-name .

   # Specific example
   docker build --progress=plain -f Dockerfile -t basinpoc .
   ```

2. Run the container with sample parameters:
   ```bash
   # Generic example
   docker run --name your-module-name \
     -e SEED=42 \
     -e NUM_CONTRACTS=5 \
     -e TOKEN_STANDARD=ERC20 \
     -e OMP_NUM_THREADS=1 \
     -e MALLOC_ARENA_MAX=2 \
     your-module-name

   # Specific example
   docker run --name basinpoc \
     -e SEED=42 \
     -e NUM_CONTRACTS=5 \
     -e TOKEN_STANDARD=ERC20 \
     -e OMP_NUM_THREADS=1 \
     -e MALLOC_ARENA_MAX=2 \
     basinpoc
   ```

### Push to Docker Hub

1. Build and push using the provided script:
   ```bash
   # Generic example
   bash ./scripts/build.sh your-dockerhub-username your-access-token

   # Specific example
   bash ./scripts/build.sh username token
   ```

### Verify Remote Image

1. Run the pushed image locally without network access:
   ```bash
   # Generic example
   docker run your-dockerhub-username/your-module-name:tag --network none

   # Specific example
   docker run mavericb/test:synth-data-v20240905193355 --network none
   ```

### Lilypad Integration

1. Update the Lilypad template:
   - Modify the `Image` field:
     ```json
     // Generic example
     "Image": "your-dockerhub-username/your-module-name:tag"

     // Specific example
     "Image": "mavericb/test:synth-data-v20240905193355"
     ```

2. Test on local Lilypad network:
   - Ensure local testnet is running (see below)
   - Execute the module:
     ```bash
     # Generic example
     ./stack run --network dev github.com/your-username/your-repo:commit-hash -i Input=test-input

     # Specific example
     ./stack run --network dev github.com/mavericb/lilypad-module-test-synth-data:a15bb98a877e816497eb08bb76ccdcd0a46efeed -i Input=moo
     ```

## Setting Up Local Lilypad Testnet

### Start Local Testnet

Run these commands in order:

```bash
./stack chain-clean
./stack chain
./stack chain-boot
ipfs daemon
./stack solver
./stack job-creator
rm -rf /home/lily/.bacalhau
./stack bacalhau-node
./stack resource-provider --offer-gpu 1
```

### Verify Local Testnet

Run these test commands:

```bash
./stack run cowsay:v0.0.4 -i Message=moo
./stack run --network dev github.com/arsenum/GPU:main
./stack run --network dev github.com/mavericb/lilypad-module-test:36fc663dde73cbc536e71020537d0e1cf49b164d -i Input=moo
```

