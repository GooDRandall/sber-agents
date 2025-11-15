import logging
import tempfile
import os
from pathlib import Path
from typing import Optional
import paramiko
from paramiko import RSAKey, Ed25519Key, ECDSAKey
from paramiko.ssh_exception import SSHException
from config import config

logger = logging.getLogger(__name__)

# Глобальная переменная для кэширования SSH клиента
_ssh_client: Optional[paramiko.SSHClient] = None


def _get_ssh_client() -> paramiko.SSHClient:
    """Создает и возвращает SSH клиент для подключения к серверу с Whisper."""
    global _ssh_client
    
    if _ssh_client is not None:
        try:
            # Проверяем, что соединение еще активно
            _ssh_client.exec_command("echo test", timeout=5)
            return _ssh_client
        except Exception:
            # Соединение разорвано, пересоздаем
            _ssh_client = None
    
    if _ssh_client is None:
        logger.info(f"Connecting to Whisper server: {config.SSH_USER}@{config.SSH_HOST}:{config.SSH_PORT}")
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        try:
            # Пробуем подключиться с указанным ключом
            key_path = Path(config.SSH_KEY_FILE)
            # Если путь относительный, пробуем расширить через expanduser
            if not key_path.is_absolute():
                key_path = key_path.expanduser()
            if not key_path.exists():
                raise FileNotFoundError(f"SSH key not found: {key_path}")
            
            logger.debug(f"Using SSH key: {key_path}")
            
            # Пробуем загрузить ключ вручную для поддержки разных форматов
            pkey = None
            key_types = [
                (RSAKey, 'RSA'),
                (Ed25519Key, 'Ed25519'),
                (ECDSAKey, 'ECDSA'),
            ]
            
            for key_class, key_type_name in key_types:
                try:
                    pkey = key_class.from_private_key_file(str(key_path))
                    logger.debug(f"Key loaded as {key_type_name}")
                    break
                except (SSHException, ValueError, IOError):
                    try:
                        with open(key_path, 'r') as f:
                            pkey = key_class.from_private_key(f, password=None)
                        logger.debug(f"Key loaded as {key_type_name} (no password)")
                        break
                    except:
                        continue
            
            if not pkey:
                raise ValueError(f"Could not load SSH key from {key_path}")
            
            client.connect(
                hostname=config.SSH_HOST,
                port=config.SSH_PORT,
                username=config.SSH_USER,
                pkey=pkey,
                timeout=30
            )
            
            logger.info(f"Successfully connected to Whisper server")
            _ssh_client = client
            return _ssh_client
            
        except Exception as e:
            logger.error(f"Failed to connect to Whisper server: {e}")
            if client:
                client.close()
            raise


def transcribe_audio(audio_path: str | Path) -> str:
    """
    Транскрибирует аудиофайл через Whisper на удаленном сервере.
    
    Args:
        audio_path: Путь к локальному аудиофайлу
        
    Returns:
        Транскрибированный текст
    """
    audio_path = Path(audio_path)
    
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    logger.info(f"Transcribing audio file: {audio_path} (size: {audio_path.stat().st_size} bytes)")
    
    try:
        # Получаем SSH клиент
        client = _get_ssh_client()
        
        # Создаем временный файл на сервере
        remote_temp_file = f"/tmp/whisper_audio_{os.getpid()}_{os.urandom(4).hex()}.ogg"
        
        # Загружаем файл на сервер
        logger.debug(f"Uploading audio file to server: {remote_temp_file}")
        sftp = client.open_sftp()
        try:
            sftp.put(str(audio_path), remote_temp_file)
            sftp.chmod(remote_temp_file, 0o644)
        finally:
            sftp.close()
        
        # Выполняем транскрибацию на сервере
        logger.info(f"Running Whisper transcription on server (model: {config.WHISPER_MODEL})...")
        
        # Команда для транскрибации через виртуальное окружение
        # Экранируем кавычки в пути к файлу
        escaped_file = remote_temp_file.replace("'", "'\\''")
        transcribe_command = f"""bash -c "source ~/whisper-env/bin/activate && python3 -c \\"
import whisper
import sys

try:
    model = whisper.load_model('{config.WHISPER_MODEL}')
    result = model.transcribe('{escaped_file}')
    print(result['text'].strip())
except Exception as e:
    print(f'ERROR: {{e}}', file=sys.stderr)
    sys.exit(1)
\\""
"""
        
        stdin, stdout, stderr = client.exec_command(transcribe_command, timeout=300)
        
        # Читаем вывод
        exit_status = stdout.channel.recv_exit_status()
        output = stdout.read().decode('utf-8').strip()
        error = stderr.read().decode('utf-8').strip()
        
        # Удаляем временный файл с сервера
        try:
            client.exec_command(f"rm -f {remote_temp_file}", timeout=5)
        except:
            pass
        
        if exit_status != 0:
            error_msg = error or "Unknown error"
            logger.error(f"Whisper transcription failed: {error_msg}")
            raise RuntimeError(f"Whisper transcription failed: {error_msg}")
        
        if not output:
            logger.warning("Whisper returned empty transcription")
            return ""
        
        logger.info(f"Transcription successful: {output[:100]}...")
        return output
        
    except paramiko.SSHException as e:
        logger.error(f"SSH error during transcription: {e}", exc_info=True)
        raise RuntimeError(f"SSH connection error: {e}")
    except Exception as e:
        logger.error(f"Error during transcription: {e}", exc_info=True)
        raise


def close_ssh_connection():
    """Закрывает SSH соединение (для использования при завершении работы)."""
    global _ssh_client
    if _ssh_client:
        try:
            _ssh_client.close()
        except:
            pass
        _ssh_client = None

