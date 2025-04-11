import asyncio
import websockets
import json

async def test_websocket():
    uri = "ws://localhost:8765"  # WebSocket server address

    async with websockets.connect(uri) as websocket:
        # Define the test query payload
        query_payload = {
            "query": "Show me all courses"
        }

        # Send the query to the WebSocket server
        await websocket.send(json.dumps(query_payload))

        # Wait for the response
        response = await websocket.recv()
        print("Response from server:", json.loads(response))

# Run the test
asyncio.run(test_websocket())