# built-in libraries
from typing import TypeVar, Any

# litellm libraries
import litellm
from litellm.types.utils import ModelResponse, Message
from litellm import completion
from instructor import from_litellm, Mode

# misc libraries
from pydantic import BaseModel, create_model

class BaseResponse(BaseModel):
    """A default response model that stores a list of predicted Ekman emotions. We will use this to predict the emotions of a review."""
    answer: str

ResponseType = TypeVar('ResponseType', bound=BaseModel)

class LLMCaller:
    """
    A class to interact with a Large Language Model (LLM)
    using the LiteLLM and Instructor libraries.
    
    Designed to send prompts and receive structured responses
    as Pydantic models (e.g., predicted emotions).
    """
    def __init__(self, api_key: str, project_id: str, api_url: str, model_id: str, params: dict[str, Any]):
        """Initializes the LLMCaller with Watsonx credentials and configuration."""
        """Args:
            api_key (str): The API key for authenticating with the LLM service.
            project_id (str): The project ID associated with the LLM service.
            api_url (str): The base URL for the LLM service API.
            model_id (str): The identifier of the specific LLM model to use.
            params (dict[str, Any]): Additional parameters to configure the LLM's behavior.
        """
        self.api_key = api_key
        self.project_id = project_id
        self.api_url = api_url
        self.model_id = model_id
        self.params = params

        litellm.drop_params = True
        self.client = from_litellm(completion, mode=Mode.JSON)

    def create_response_model(self, title: str, fields: dict) -> ResponseType:
        """Dynamically creates a Pydantic response model for the LLM's output.
        Args:
            title (str): The name of the response model.
            fields (dict): A dictionary defining the fields of the response model.
                           Keys are field names, and values are tuples of (type, Field).

        Returns:
            ResponseType: A dynamically created Pydantic model class.
        """
        return create_model(title, **fields, __base__=BaseResponse)

    def invoke(self, prompt: str, response_model: ResponseType = BaseResponse, **kwargs) -> ResponseType:
        """Sends a prompt to the LLM and retrieves a structured response.

        Args:
            prompt (str): The input prompt to send to the LLM.
            response_model (ResponseType): The Pydantic model to structure the LLM's response.
                                           Defaults to BaseResponse.
            **kwargs: Additional arguments to pass to the LLM client.

        Returns:
            ResponseType: The structured response from the LLM, parsed into the specified response model.
        """
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": prompt + "\n\nRespond using this structure: " + str(response_model.__annotations__)
            }],
            project_id=self.project_id,
            apikey=self.api_key,
            api_base=self.api_url,
            response_model=response_model,
            **kwargs
        )
        return response
    
    def chat(self, messages: list[dict[str, str] | Message], **kwargs) -> ModelResponse:
        """Sends a prompt to the LLM without a structured response. I.e. we don't use
        instructor to parse the response in this case."""

        return completion(
            model=self.model_id,
            project_id=self.project_id,
            apikey=self.api_key,
            api_base=self.api_url,
            messages=messages,
            **kwargs,
        )
