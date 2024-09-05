## Steps to turn on local lilypad via Ultra Large Dell node

### Start local testnet

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

### Test local testnet

```bash
./stack run cowsay:v0.0.4 -i Message=moo
./stack run --network dev github.com/arsenum/GPU:main
./stack run --network dev github.com/mavericb/lilypad-module-test:36fc663dde73cbc536e71020537d0e1cf49b164d -i Input=moo
```

### Development guide for a new module

1. Update Dockerfile
   - Download necessary files for build
   - Point to downloaded files (runtime won't have internet access)

2. Update `main.py`
   - Add your code
   - Access input through environment variable:
     ```python
     input = os.environ.get("INPUT") or "question mark floating in space"
     ```

3. Update `requirements.txt` with your requirements

4. Keep `start.sh` and `build.sh` as they are

5. Update `config.env` with your image name and tag, for example:
   ```
   # Image details
   IMAGE_NAME=test
   IMAGE_TAG=synth-data
   ```

### Test the new image

1. Test in Docker locally:
   ```bash
   docker build --progress=plain -f Dockerfile -t basinpoc .
   docker run --name basinpoc -e SEED=42 -e NUM_CONTRACTS=5 -e TOKEN_STANDARD=ERC20 -e OMP_NUM_THREADS=1 -e MALLOC_ARENA_MAX=2 basinpoc
   ```

2. Push the image to Docker Hub:
   ```bash
   bash ./scripts/build.sh username token
   ```

3. Test the remote image locally:
   ```bash
   docker run mavericb/test:synth-data-v20240905193355 --network none
   ```

4. Update template of lilypad and run on local test:
   - Update the Image field in the template:
     ```json
     "Image": "mavericb/test:synth-data-v20240905193355"
     ```
   - Run the test:
     ```bash
     ./stack run --network dev github.com/mavericb/lilypad-module-test-synth-data:a15bb98a877e816497eb08bb76ccdcd0a46efeed -i Input=moo
     ```

