from PIL import Image
from azure.identity import get_bearer_token_provider, AzureCliCredential
from io import BytesIO
from openai import AzureOpenAI, RateLimitError, AuthenticationError
import base64
import subprocess 
import time 

###### Models

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(BytesIO(image_data))
    if image.mode in ('RGBA', 'P'):
        image = image.convert('RGB')
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def base64encode(query_image):
    buffered = BytesIO()
    if query_image.mode in ('RGBA', 'P'):
        query_image = query_image.convert('RGB')
    query_image.save(buffered, format="JPEG")
    base64_bytes = base64.b64encode(buffered.getvalue())
    base64_string = base64_bytes.decode("utf-8")
    return base64_string

class GPTEndPoint:
    def __init__(
        self, 
        model_name: str, 
        logger,
        sys_prompt: str = "You are a data generation agent.\
            Given example image, question-answer pairs and a new image,\
            generate a single question and answer similar to the examples.", 
        max_retries=5,
        retry_delay_sec=2,
        pim_retry_delay_sec=120,
    ):
        
        supported_model_names = ['gpt-4o', 'gpt-4o-450K', 'gpt-4-july', 'gpt-4o-australia-east', 'gpt-4o-australia-east-2', 'mock']
        assert model_name in supported_model_names, "model_name must be in "+', '.join(supported_model_names)
        self.model_name = model_name
        self.azure_endpoint_url = "https://openai-models-west-us3.openai.azure.com/"
        if "australia" in self.model_name:
            self.azure_endpoint_url = "https://openai-models-australia-east.openai.azure.com/"
        self.sys_prompt = sys_prompt
        self.logger = logger
        self.max_retries = max_retries
        self.retry_delay_sec = retry_delay_sec
        self.pim_retry_delay_sec = pim_retry_delay_sec
        
        if model_name == "mock":
            self.logger.debug("Mocking GPT4 API")
            for attr, value in self.__dict__.items():
                self.logger.debug(f"{attr} = {value}")
            return
        token_provider = get_bearer_token_provider(
            AzureCliCredential(), "https://cognitiveservices.azure.com/.default"
        )
        
        self.client = AzureOpenAI(
            api_version="2023-06-01-preview",
            azure_endpoint=self.azure_endpoint_url,
            azure_ad_token_provider=token_provider
        )
        
        self.logger.debug(f"GPTEndPoint created with model_name: {self.model_name}")
        self.logger.debug(f"System prompt: {self.sys_prompt}")
        self.logger.debug("Debug Mode On")

    def create_request(self, contents, is_base64=False):      
        messages = []
        if self.sys_prompt:
            messages.append({"role": "system", "content": self.sys_prompt})
            
        user_content = {"role": "user", "content": []}
        for i, content in enumerate(contents):
            if type(content) == str:
                self.logger.debug(f"GPT4 Content {i}: type=text; text={content}")
                user_content["content"].append(
                    {
                        "type": "text",
                        "text": content
                    }
                )
            else:
                self.logger.debug(f"GPT4 Content {i}: type=image")
                encoded_image = content
                if not is_base64:
                    encoded_image = base64encode(content)
                user_content["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpg;base64,{encoded_image}",
                        },  
                    }
                )
        
        # Generate           
        messages.append(user_content)
        
        request = {"messages": messages}
        
        self.logger.debug(f"Created request")
        
        # # Overlogging for debugging
        # for message in messages:
        #     self.logger.debug("Role: " + message["role"])
        #     if type(message["content"]) == str:
        #         self.logger.debug("Content: single_str")
        #     else:
        #         for content in message["content"]:
        #             self.logger.debug("type: " + content["type"])
        #             self.logger.debug("keys: " + str(list(content.keys())))

        return request

    def get_response(self, request):
        for attempt in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    **request,
                )
                openai_response = completion.model_dump()
                
                self.logger.debug(f"Received response: {openai_response}")
                
                return openai_response["choices"][0]["message"]["content"]
            
            except RateLimitError as e:
                if attempt + 1 < self.max_retries:
                    self.logger.warning(f"Rate limit hit: {e}. Attempt {attempt + 1} of {self.max_retries}. Retrying in {self.retry_delay_sec} seconds...")
                    time.sleep(self.retry_delay_sec * (attempt + 1))
                else:
                    self.logger.error("Max retries reached. Raising RateLimitError.")
                    raise e
            except AuthenticationError as e:
                if attempt + 1 < self.max_retries:
                    self.logger.warning(f"Authentication Error Hit: {e}. Attempt {attempt + 1} of {self.max_retries}. Programmatic PIM")
                    time.sleep(self.pim_retry_delay_sec)
                    stdout, stderr, status_code = self.pim()
                    self.logger.warning(f"Status Code: {status_code} \nOutput: {stdout}\n{stderr}")
                    self.logger.warning(f"Retrying in {self.pim_retry_delay_sec} seconds...")
                else:
                    self.logger.error("Max retries reached. Raising AuthenticationError.")
                    raise e
                
    def pim(self):
        command = ["pimtool", "-c", "/home/t-sijoshi/multimodal-data-gen/gpt4_api_pim.json", "-y"]

        # Run the command
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Extract the stdout, stderr, and return code
        stdout = result.stdout.decode('utf-8')
        stderr = result.stderr.decode('utf-8')
        return_code = result.returncode
        
        return stdout, stderr, return_code
    
    def generate(self, contents, is_base64:bool=False):
        if self.model_name == "mock":
            self.logger.debug("Mock Request")
            for content in contents:
                self.logger.debug(content)
            return "Mock Response"
        msgs = self.create_request(contents, is_base64)
        response = self.get_response(msgs)
        self.logger.debug(f"Generated response: {response}")
        return response
