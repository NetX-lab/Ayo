import re


def print_warning(message):
    # yellow color
    print(f"\033[93m{message}\033[0m")


def print_key_info(message):
    # green color
    print(f"\033[92m{message}\033[0m")


def print_error(message):
    # red color
    print(f"\033[91m{message}\033[0m")


def format_query_expanding_prompt(question, prompt_template, expanded_query_num=3):
    keys = ", ".join([f"question{i+1}" for i in range(expanded_query_num)])
    json_example = (
        "{\n      "
        + "\n      ".join(
            [
                f'"question{i+1}": "[refined version {i+1}]"'
                + ("," if i < expanded_query_num - 1 else "")
                for i in range(expanded_query_num)
            ]
        )
        + "\n    }"
    )

    return prompt_template.format(
        expanded_query_num=expanded_query_num,
        question=question,
        keys=keys,
        json_example=json_example,
    )


def rename_template_placeholders(template, placeholder_mapping):
    """
    Modify the placeholder names in the template string

    Args:
        template (str): The template string containing placeholders
        placeholder_mapping (dict): The placeholder mapping, format as {old_name: new_name}

    Returns:
        str: The modified template string
    """
    result = template
    for old_name, new_name in placeholder_mapping.items():
        result = result.replace(f"{{{old_name}}}", f"{{{new_name}}}")
    return result


def fill_prompt_template_with_placeholdersname_approximations(
    prompt_template, input_kwargs
):
    """
    Fill the prompt template, handle the mismatch between placeholder names and input parameter names

    Args:
        prompt_template (str): The prompt template containing placeholders
        input_kwargs (dict): The input parameters dictionary

    Returns:
        str: The filled prompt template
    """
    import re
    from difflib import get_close_matches

    # find all placeholders in the prompt template
    placeholders = re.findall(r"\{([^{}]+)\}", prompt_template)
    result = prompt_template

    # create a match cache to avoid duplicate calculations
    input_keys = list(input_kwargs.keys())

    # find the best match for each placeholder
    for placeholder in placeholders:
        if placeholder in input_kwargs:
            # if there is an exact match, use it
            value = str(input_kwargs[placeholder])
            result = result.replace(f"{{{placeholder}}}", value)
        else:
            # check if the match result is already in the cache
            close_matches = get_close_matches(placeholder, input_keys, n=1, cutoff=0.1)

            if close_matches:
                print_warning(
                    f"Use approximate matching: Placeholder '{placeholder}' is matched to input parameter '{close_matches[0]}'"
                )
                value = str(input_kwargs[close_matches[0]])
                result = result.replace(f"{{{placeholder}}}", value)
            else:
                print_error(f"No match found for placeholder '{placeholder}'")

    return result


def check_unfilled_placeholders_in_prompt_template(prompt_template):
    """
    Check if the prompt template is complete by ensuring all placeholders are filled (no {placeholder} pattern in the prompt template)
    """
    placeholders = re.findall(r"\{([^{}]+)\}", prompt_template)
    if placeholders:
        raise ValueError(
            "Prompt template is not complete without any unfilled placeholders"
        )
    else:
        return True


def check_prompt_template_and_placeholders_match(prompt_template, input_kwargs):
    """
    Check if the prompt template is complete by ensuring all placeholders are filled

    Args:
        prompt_template (str): The prompt template to check
        input_kwargs (dict): The input kwargs to check

    Returns:
        bool: True if the prompt template is complete, False otherwise
    """

    # check if all placeholders in the prompt template are in the input_kwargs
    import re

    placeholders = re.findall(r"\{([^{}]+)\}", prompt_template)
    for placeholder in placeholders:
        if placeholder not in input_kwargs:
            raise ValueError(f"Placeholder {placeholder} not found in input_kwargs")

    return True


def fill_prompt_template(prompt_template, input_kwargs):
    """
    Fill the placeholders in the prompt template with the input kwargs
    """
    for key, value in input_kwargs.items():
        prompt_template = prompt_template.replace(f"{{{key}}}", value)

    # check if the prompt template is complete
    return prompt_template
