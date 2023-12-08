from typing import Any, List, Optional, Union

from langchain.llms.vertexai import VertexAIModelGarden
from langchain.schema.output import Generation, LLMResult


class VertexAIModelGardenPeft(VertexAIModelGarden):
    """
    A class representing large language models served from Vertex AI Model Garden using Peft
    PyTorch runtime such as LLaMa2.

    Attributes:
        max_length (int): Token limit determines the maximum amount of text output from one prompt.
        top_k (int): How the model selects tokens for output, the next token is selected from
            among the top-k most probable tokens. Top-k is ignored for Code models.
    """

    max_length: int = 200
    top_k: int = 40

    def __init__(self, **kwargs):
        """
        Initialize the VertexAIModelGardenPeft instance.

        Args:
            **kwargs: Additional keyword arguments to be passed to the super class.
        """
        super().__init__(
            allowed_model_args=[
                "max_length",
                "top_k",
            ],
            **kwargs,
        )

    def _generate(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = super()._generate(
            prompts=prompts,
            max_length=self.max_length,
            top_k=self.top_k,
            **kwargs,
        )
        return LLMResult(
            generations=_normalize_generations(prompts, result.generations)
        )

    async def _agenerate(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = await super()._agenerate(
            prompts=prompts,
            max_length=self.max_length,
            top_k=self.top_k,
            **kwargs,
        )
        return LLMResult(
            generations=_normalize_generations(prompts, result.generations)
        )


# TODO: Support VLLM streaming inference
class VertexAIModelGardenVllm(VertexAIModelGarden):
    """
    A class representing large language models served from Vertex AI Model Garden using VLLM
    PyTorch runtime such as LLaMa2.

    Attributes:
        max_length (int): Token limit determines the maximum amount of text output from one prompt.
        top_k (int): How the model selects tokens for output, the next token is selected from
            among the top-k most probable tokens. Top-k is ignored for Code models.
        n (int): Number of output sequences to return for the given prompt.
        best_of (int): Number of output sequences that are generated from the prompt.
            From these `best_of` sequences, the top `n` sequences are returned.
            `best_of` must be greater than or equal to `n`. This is treated as
            the beam width when `use_beam_search` is True. By default, `best_of`
            is set to `n`.
        presence_penalty (float): Float that penalizes new tokens based on whether they
            appear in the generated text so far. Values > 0 encourage the model
            to use new tokens, while values < 0 encourage the model to repeat
            tokens.
        frequency_penalty (float): Float that penalizes new tokens based on their
            frequency in the generated text so far. Values > 0 encourage the
            model to use new tokens, while values < 0 encourage the model to
            repeat tokens.
        repetition_penalty (float): Float that penalizes new tokens based on whether
            they appear in the prompt and the generated text so far. Values > 1
            encourage the model to use new tokens, while values < 1 encourage
            the model to repeat tokens.
        temperature (float): Float that controls the randomness of the sampling. Lower
            values make the model more deterministic, while higher values make
            the model more random. Zero means greedy sampling.
        top_p (float): Float that controls the cumulative probability of the top tokens
            to consider. Must be in (0, 1]. Set to 1 to consider all tokens.
        top_k (int): Integer that controls the number of top tokens to consider. Set
            to -1 to consider all tokens.
        min_p (float): Float that represents the minimum probability for a token to be
            considered, relative to the probability of the most likely token.
            Must be in [0, 1]. Set to 0 to disable this.
        use_beam_search (bool): Whether to use beam search instead of sampling.
        length_penalty (float): Float that penalizes sequences based on their length.
            Used in beam search.
        early_stopping (Union[bool, str]): Controls the stopping condition for beam search. It
            accepts the following values: `True`, where the generation stops as
            soon as there are `best_of` complete candidates; `False`, where an
            heuristic is applied and the generation stops when is it very
            unlikely to find better candidates; `"never"`, where the beam search
            procedure only stops when there cannot be better candidates
            (canonical beam search algorithm).
        stop (Optional[List[str]]): List of strings that stop the generation when they are generated.
            The returned output will not contain the stop strings.
        stop_token_ids (Optional[List[int]]): List of tokens that stop the generation when they are
            generated. The returned output will contain the stop tokens unless
            the stop tokens are special tokens.
        ignore_eos (bool): Whether to ignore the EOS token and continue generating
            tokens after the EOS token is generated.
        max_tokens (Optional[int]): Maximum number of tokens to generate per output sequence.
        logprobs (Optional[int]): Number of log probabilities to return per output token.
            Note that the implementation follows the OpenAI API: The return
            result includes the log probabilities on the `logprobs` most likely
            tokens, as well the chosen tokens. The API will always return the
            log probability of the sampled token, so there  may be up to
            `logprobs+1` elements in the response.
        prompt_logprobs (Optional[int]): Number of log probabilities to return per prompt token.
        skip_special_tokens (Optional[bool]): Whether to skip special tokens in the output.
        spaces_between_special_tokens (Optional[bool]): Whether to add spaces between special
            tokens in the output.  Defaults to True.
    """

    max_length: int = 200
    top_k: int = 40
    n: int = 1
    best_of: int = 1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    use_beam_search: bool = False
    length_penalty: float = 1.0
    early_stopping: Union[bool, str] = True
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = None
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None
    skip_special_tokens: Optional[bool] = None
    spaces_between_special_tokens: Optional[bool] = True

    def __init__(self, **kwargs):
        """
        Initialize the VertexAIModelGardenVllm instance.

        Args:
            **kwargs: Additional keyword arguments to be passed to the super class.
        """
        super().__init__(
            allowed_model_args=[
                "max_length",
                "top_k",
                "n",
                "best_of",
                "presence_penalty",
                "frequency_penalty",
                "repetition_penalty",
                "temperature",
                "top_p",
                "top_k",
                "min_p",
                "use_beam_search",
                "length_penalty",
                "early_stopping",
                "stop",
                "stop_token_ids",
                "ignore_eos",
                "max_tokens",
                "logprobs",
                "prompt_logprobs",
                "skip_special_tokens",
                "spaces_between_special_tokens",
            ],
            **kwargs,
        )

    def _generate(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = super()._generate(
            prompts=prompts,
            max_length=self.max_length,
            top_k=self.top_k,
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            use_beam_search=self.use_beam_search,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            **kwargs,
        )
        return LLMResult(
            generations=_normalize_generations(prompts, result.generations)
        )

    async def _agenerate(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> LLMResult:
        """
        Generate text based on the given prompts.

        Args:
            prompts (List[str]): List of prompts to generate text from.
            **kwargs: Additional keyword arguments.

        Returns:
            LLMResult: The generated text.

        """
        result = await super()._agenerate(
            prompts=prompts,
            max_length=self.max_length,
            top_k=self.top_k,
            n=self.n,
            best_of=self.best_of,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            temperature=self.temperature,
            top_p=self.top_p,
            min_p=self.min_p,
            use_beam_search=self.use_beam_search,
            length_penalty=self.length_penalty,
            early_stopping=self.early_stopping,
            stop=self.stop,
            stop_token_ids=self.stop_token_ids,
            ignore_eos=self.ignore_eos,
            max_tokens=self.max_tokens,
            logprobs=self.logprobs,
            prompt_logprobs=self.prompt_logprobs,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
            **kwargs,
        )
        return LLMResult(
            generations=_normalize_generations(prompts, result.generations)
        )


def _normalize_generations(
    prompts: List[str], generations: List[List[Generation]]
) -> List[List[Generation]]:
    return [
        [Generation(text=_normalize_text(result_generations))]
        for prompt, result_generations in zip(prompts, generations)
    ]


def _normalize_text(generations: List[Generation]) -> str:
    text = map(lambda x: x.text, generations)
    text = "".join(text)
    text = text[text.find("Output:") :]
    text = text[len("Output:") :]
    text = text.strip()
    return text
