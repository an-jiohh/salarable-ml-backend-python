from app.core.config import Settings, get_settings
from requests import Session
import httpx
import json
import time
import io
import traceback
import asyncio

class RTZRClient:
    def __init__(self, config:Settings):
        super().__init__()
        self.api_url = config.rtzr_api_url
        self.client_id = config.rtzr_client_id
        self.client_secret = config.rtzr_client_secret
        self._sess = Session()
        self._token = None
        self.polling_interval = 1

    @property
    def token(self):
        if self._token is None or self._token["expire_at"] < time.time():
            resp = self._sess.post(
                self.api_url + "/v1/authenticate",
                data={"client_id": self.client_id, "client_secret": self.client_secret},
            )
            resp.raise_for_status()
            self._token = resp.json()
        return self._token["access_token"]
    
    async def send_audio_file(self, files):
        print("send_audio_file")
        try : 
            async with httpx.AsyncClient() as client:
                config = {
                    "use_diarization": True,
                    "diarization": {
                        "spk_count": 1
                    },
                    "use_itn": False,
                    "use_disfluency_filter": False,
                    "use_profanity_filter": False,
                    "use_paragraph_splitter": True,
                    "paragraph_splitter": {
                        "max": 50
                    }
                }
                headers = {'Authorization': 'bearer ' + self.token}
                data = {'config': json.dumps(config)}

                print("file open")

                try :  
                    resp = await client.post(
                    self.api_url + "/v1/transcribe",
                    headers=headers,
                    data=data,
                    files={'file':io.BytesIO(files)},
                    )
                    resp.raise_for_status()
                    return resp.json()["id"]
                except httpx.HTTPStatusError as e:
                        print(f"HTTP 에러 발생: {e}")
                        print(f"응답 본문: {e.response.text}")
                except Exception as e:
                        print(f"예기치 않은 오류 발생: {e}")
                        traceback.print_exc()
        except Exception as e:
            print(e)
            traceback.print_exc()

    async def poll_stt_status(self, TRANSCRIBE_ID):
        headers = {'Authorization': 'bearer ' + self.token}
        while True:
            print(1)
            try: 
                async with httpx.AsyncClient() as client:
                    try :  
                        resp = await client.get(
                            self.api_url + '/v1/transcribe/'+f'{TRANSCRIBE_ID}',
                            headers=headers,
                        )
                        resp.raise_for_status()
                        response = resp.json()
                        if response["status"] == "completed":
                            trnascription = ""
                            for result in response["results"]['utterances']:
                                trnascription += result['msg']
                            return trnascription
                        else:
                            print(response)
                    except httpx.HTTPStatusError as e:
                            print(f"HTTP 에러 발생: {e}")
                            print(f"응답 본문: {e.response.text}")
                            raise
                    except Exception as e:
                            print(f"예기치 않은 오류 발생: {e}")
                await asyncio.sleep(self.polling_interval)
            except Exception as e:
                print(e)
                traceback.print_exc()

rtzr_client = RTZRClient(config=get_settings())

def get_rtzr_client() :
    yield rtzr_client