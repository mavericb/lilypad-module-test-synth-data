# Running the Synthetic Data Generation Module

This guide provides instructions for setting up, testing, and running the synthetic data generation module on Lilypad.

## Setup and Configuration

### 1. Dockerfile
- Ensure all necessary files for the build are downloaded and referenced.
- The runtime environment won't have internet access, so all dependencies must be included.

### 2. main.py
- The core logic for synthetic data generation is implemented here.
- Input is accessed through environment variables:
  ```python
  input = os.environ.get("INPUT") or "question mark floating in space"
  ```

### 3. requirements.txt
- Lists all Python package dependencies for the module.

### 4. config.env
- Contains image details:
  ```
  IMAGE_NAME=test
  IMAGE_TAG=synth-data
  ```

### 5. start.sh and build.sh
- These scripts handle standard operations and should not be modified.

## Testing and Deployment

### Local Docker Testing

1. Build the Docker image:
   ```bash
   docker build --progress=plain -f Dockerfile -t basinpoc .
   ```

2. Run the container:
   ```bash
   docker run --name basinpoc -e SEED=42 -e NUM_CONTRACTS=5 -e TOKEN_STANDARD=ERC20 -e OMP_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2 basinpoc
   ```

### Push to Docker Hub

1. Build and push the image:
   ```bash
   bash ./scripts/build.sh username token
   ```

### Verify Remote Image

1. Run the pushed image locally without network access:
   ```bash
   docker run mavericb/test:synth-data-v20240905193355 --network none
   ```

### Lilypad Integration

1. Update the Lilypad template:
   - Modify the `Image` field in your Lilypad JSON template:
     ```json
     "Image": "mavericb/test:synth-data-v20240905193355"
     ```

2. Test on local Lilypad network:
   - Ensure local testnet is running (see below)
   - Execute the module:
     ```bash
     ./stack run --network dev github.com/mavericb/lilypad-module-test-synth-data:a15bb98a877e816497eb08bb76ccdcd0a46efeed -i Input=moo
     ```

## Setting Up Local Lilypad Testnet

### Start Local Testnet

Run each of these commands in a separate terminal window or tab, in the order listed:

```bash
# Terminal 1
./stack chain-clean
./stack chain

# Terminal 2
./stack chain-boot

# Terminal 3
ipfs daemon

# Terminal 4
./stack solver

# Terminal 5
./stack job-creator

# Terminal 6
rm -rf /home/lily/.bacalhau
./stack bacalhau-node

# Terminal 7
./stack resource-provider --offer-gpu 1
```

Keep all terminals open as these processes need to run concurrently for the local testnet to function properly.

### Verify Local Testnet

Once all the above processes are running, open a new terminal to run these test commands:

```bash
./stack run cowsay:v0.0.4 -i Message=moo
./stack run --network dev github.com/arsenum/GPU:main
./stack run --network dev github.com/mavericb/lilypad-module-test:36fc663dde73cbc536e71020537d0e1cf49b164d -i Input=moo
```
