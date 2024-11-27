import json

def lambda_handler(event, context):
    """
    Langchain Lambda Moderator
    """
    return {
        "statusCode": 200,
        "body": json.dumps({
            "message": "hello world",
            # "location": ip.text.replace("\n", "")
        }),
    }
