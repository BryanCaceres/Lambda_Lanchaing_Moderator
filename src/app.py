import json
from src.services.agent_general_moderation import GeneralModeration
import traceback

general_moderator = GeneralModeration()

def lambda_handler(event, context):
    """
    Langchain Lambda Moderator
    """
    try:
        request_body = json.loads(event["body"])
        input_text = request_body.get("input_text", "")

        model_response = general_moderator.moderate(input_text)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "model_response": model_response
            })
        }
    except Exception as e:
        traceback.print_exc()
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
