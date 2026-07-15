"""
Binary and Model Setup Downloader for ZovaAI.
Downloads and verifies Whisper models, Piper voice synthesizers, and voice models.
Supports range-based download resumption and detailed progress tracking.
"""

import sys
import zipfile
from pathlib import Path
from typing import Optional, Tuple, Dict
import requests
from tqdm import tqdm

from src.core.config import Config
from src.core.logger import LoggerSetup, get_logger
from src.core.exceptions import SetupError

logger = get_logger("setup_binaries")


class BinarySetupManager:
    """Manages directory setup and downloads for AI models and binaries."""

    def __init__(self, config: Config):
        """
        Initializes the setup manager.
        
        Args:
            config: An initialized Config manager instance.
        """
        self.config = config

    def ensure_directories(self) -> None:
        """Creates target directories for models, binaries, and temporary files."""
        dirs_to_create = [
            self.config.speech_recognition.model_dir,
            self.config.speech_synthesis.model_path.parent,
            self.config.speech_synthesis.executable_path.parent,
            self.config.speech_synthesis.output_dir,
            self.config.project_root / "temp"
        ]
        
        for directory in dirs_to_create:
            try:
                if not directory.exists():
                    logger.info("Creating directory: %s", directory)
                    directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise SetupError(
                    f"Failed to create directory '{directory}': {str(e)}"
                ) from e

    def _determine_download_state(
        self, url: str, dest_path: Path
    ) -> Tuple[int, Dict[str, str], str, Optional[int]]:
        """
        Queries HEAD request and checks filesystem size to determine the download state.
        
        Returns:
            Tuple containing:
            - start_byte (int)
            - headers (dict)
            - file_mode (str)
            - expected_size (Optional[int])
        """
        start_byte = 0
        headers: Dict[str, str] = {}
        file_mode = "wb"
        expected_size: Optional[int] = None
        supports_range = False

        # 1. Fetch remote content size and check if we support Range requests
        try:
            head_res = requests.head(url, allow_redirects=True, timeout=15)
            head_res.raise_for_status()
            
            content_length = head_res.headers.get("content-length")
            if content_length:
                expected_size = int(content_length)
                
            accept_ranges = head_res.headers.get("accept-ranges") or ""
            accept_ranges_cap = head_res.headers.get("Accept-Ranges") or ""
            supports_range = "bytes" in accept_ranges or "bytes" in accept_ranges_cap
        # pylint: disable=broad-exception-caught
        except Exception as e:
            logger.warning(
                "HEAD request failed for %s, falling back to clean download: %s",
                url, e
            )

        # 2. Check if file already exists and determine if we can resume
        if dest_path.exists() and dest_path.stat().st_size > 0:
            local_size = dest_path.stat().st_size
            
            if expected_size and local_size == expected_size:
                logger.info("File already complete: %s (skipping)", dest_path.name)
                return -1, headers, file_mode, expected_size
            
            if expected_size and local_size > expected_size:
                logger.warning(
                    "Local file size (%d) exceeds remote size (%d). Resetting: %s",
                    local_size, expected_size, dest_path.name
                )
                dest_path.unlink()
            elif supports_range:
                start_byte = local_size
                headers["Range"] = f"bytes={start_byte}-"
                file_mode = "ab"
                logger.info(
                    "Resuming download from byte %d for %s",
                    start_byte, dest_path.name
                )
            else:
                logger.info(
                    "Local file incomplete and resume not supported. Restarting: %s",
                    dest_path.name
                )
                dest_path.unlink()

        return start_byte, headers, file_mode, expected_size

    def _execute_download_stream(self, params: Dict[str, str]) -> Optional[int]:
        """Performs HTTP chunked stream reading and updates the progress bar."""
        headers_cleaned = {
            k.replace("headers_", ""): v
            for k, v in params.items() if k.startswith("headers_")
        }
        
        start_byte = int(params["start_byte"])
        file_mode = params["file_mode"]
        expected_size = int(params["expected_size"]) if params.get("expected_size") else None

        logger.info("Downloading: %s", params["url"])
        
        # If server returns 200 instead of 206 when range was requested, we reset to wb
        with requests.get(params["url"], headers=headers_cleaned, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            status_code = r.status_code
            if status_code == 200 and start_byte > 0:
                logger.warning("Server ignored Range request. Restarting download.")
                file_mode = "wb"
                start_byte = 0
            
            # If we didn't get Content-Length from HEAD, check GET response headers
            if expected_size is None:
                get_length = r.headers.get("content-length")
                if get_length:
                    expected_size = int(get_length) + start_byte
            
            total_to_download = expected_size if expected_size else start_byte
            
            # pylint: disable=unspecified-encoding
            with open(Path(params["dest_path"]), file_mode) as f:
                with tqdm(
                    total=total_to_download,
                    initial=start_byte,
                    unit="B",
                    unit_scale=True,
                    desc=params["desc"],
                    file=sys.stdout
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=1024 * 64):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
                            
        return expected_size

    def download_file(self, url: str, dest_path: Path, desc: str) -> None:
        """
        Downloads a file with support for resuming interrupted downloads.
        
        Args:
            url: The HTTP download URL.
            dest_path: Target path where the file will be saved.
            desc: Progress bar description label.
            
        Raises:
            SetupError: If the download fails or fails validation.
        """
        try:
            # 1. Fetch current download state parameters
            start_byte, headers, file_mode, expected_size = self._determine_download_state(
                url, dest_path
            )
            
            # -1 indicates download is already complete and should be skipped
            if start_byte == -1:
                return

            # Package arguments to pass to the stream worker
            params = {
                "url": url,
                "dest_path": str(dest_path),
                "file_mode": file_mode,
                "start_byte": str(start_byte),
                "expected_size": str(expected_size) if expected_size else "",
                "desc": desc
            }
            # Add headers with prefix to avoid collision
            for k, v in headers.items():
                params[f"headers_{k}"] = v

            # 2. Run the chunked download stream
            expected_size = self._execute_download_stream(params)
            
            # 3. Verify downloaded file presence and sizing
            if not dest_path.exists():
                raise SetupError(f"Downloaded file was not created: {dest_path.name}")
                
            if expected_size and dest_path.stat().st_size != expected_size:
                raise SetupError(
                    f"File size mismatch for {dest_path.name}: "
                    f"Expected {expected_size} bytes, got {dest_path.stat().st_size} bytes"
                )
                
            logger.info("Download completed and verified: %s", dest_path.name)
            
        except Exception as e:
            if not isinstance(e, SetupError):
                raise SetupError(f"Download failed for {url}: {str(e)}") from e
            raise e

    def extract_zip(self, zip_path: Path, extract_to: Path) -> None:
        """
        Extracts a ZIP archive to a destination folder.
        
        Args:
            zip_path: Path to the ZIP file.
            extract_to: Directory where the zip contents will be extracted.
            
        Raises:
            SetupError: If extraction fails.
        """
        try:
            logger.info("Extracting %s to %s", zip_path.name, extract_to)
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                members = zip_ref.namelist()
                with tqdm(total=len(members), desc="Extracting", file=sys.stdout) as pbar:
                    for member in members:
                        zip_ref.extract(member, path=extract_to)
                        pbar.update(1)
            logger.info("Extraction completed successfully.")
        except Exception as e:
            raise SetupError(f"Failed to extract {zip_path.name}: {str(e)}") from e

    def setup(self) -> None:
        """Executes the complete setup sequence for folders, binaries, and models."""
        logger.info("Starting ZovaAI setup pipeline...")
        self.ensure_directories()
        
        # 1. Download Whisper model (ggml-base.en.bin)
        whisper_dest = self.config.speech_recognition.model_dir / "ggml-base.en.bin"
        self.download_file(
            url=self.config.setup.whisper_model_url,
            dest_path=whisper_dest,
            desc="Whisper Model"
        )
        
        # 2. Download Piper voice ONNX model
        piper_voice_dest = self.config.speech_synthesis.model_path
        self.download_file(
            url=self.config.setup.piper_voice_url,
            dest_path=piper_voice_dest,
            desc="Piper Voice ONNX"
        )
        
        # 3. Download Piper voice config JSON
        piper_config_dest = self.config.speech_synthesis.config_path
        self.download_file(
            url=self.config.setup.piper_voice_config_url,
            dest_path=piper_config_dest,
            desc="Piper Voice Config"
        )
        
        # 4. Download and extract Piper executable
        piper_zip_path = self.config.project_root / "temp" / "piper_windows.zip"
        piper_bin_dir = self.config.speech_synthesis.executable_path.parent
        piper_exe_path = self.config.speech_synthesis.executable_path
        
        if not piper_exe_path.exists():
            self.download_file(
                url=self.config.setup.piper_zip_url,
                dest_path=piper_zip_path,
                desc="Piper Binaries ZIP"
            )
            
            # Extract to bin folder.
            self.extract_zip(zip_path=piper_zip_path, extract_to=piper_bin_dir.parent)
            
            # Clean up the downloaded ZIP to save space
            if piper_zip_path.exists():
                logger.info("Cleaning up temporary zip file: %s", piper_zip_path.name)
                piper_zip_path.unlink()
        else:
            logger.info("Piper executable already exists at %s (skipping)", piper_exe_path)
            
        # Verify that the final executable exists
        if not piper_exe_path.exists():
            raise SetupError(
                f"Piper executable was not found at {piper_exe_path} after extraction."
            )
            
        logger.info("=========================================")
        logger.info("ZovaAI setup completed successfully!")
        logger.info("All binaries and models are verified.")
        logger.info("=========================================")


def main() -> None:
    """Entry point for executing the setup script from the CLI."""
    try:
        # Load configuration
        config = Config()
        
        # Initialize logging
        LoggerSetup.initialize(
            log_level=config.logging.level,
            log_file=config.logging.log_file,
            max_bytes=config.logging.max_bytes,
            backup_count=config.logging.backup_count
        )
        
        manager = BinarySetupManager(config)
        manager.setup()
        
    # pylint: disable=broad-exception-caught
    except Exception as e:
        logger.error("Setup failed: %s", e)
        print(f"\nFATAL SETUP ERROR: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
