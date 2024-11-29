import json
from services.agent_general_moderation import GeneralModeration
import traceback

general_moderator = GeneralModeration()

def lambda_handler(event, context):
    """
    Langchain Lambda Moderator
    """
    try:
        request_body = json.loads(event["body"])
        comment_body = request_body.get("comment_body", "")

        model_response = general_moderator.moderate(comment_body)

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
