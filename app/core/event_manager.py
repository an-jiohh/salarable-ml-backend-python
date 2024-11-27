import asyncio
import json

# 이벤트를 저장할 Queue
event_queue = asyncio.Queue()

# SSE 스트림 함수
async def event_stream():
    while True:
        # Queue에서 딕셔너리를 가져옴
        data = await event_queue.get()
        # JSON 직렬화
        json_data = json.dumps(data)
        yield f"data: {json_data}\n\n"

# Queue에 딕셔너리를 추가하는 함수
async def add_event(data: dict):
    if 'data' in data and isinstance(data['data'], (dict, list)):
        data['data'] = json.dumps(data['data'],  ensure_ascii=False)
    await event_queue.put(data)