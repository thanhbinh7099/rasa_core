from rasa_nlu.converters import load_markdown_data


def convert(src, dst):
    with open(dst, "w") as text_file:
        text_file.write(load_markdown_data(src).as_json())
