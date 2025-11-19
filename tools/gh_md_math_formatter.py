import argparse
import os
import re
import shutil

import unicodeit


def convert_math_fonts_to_unicode(text):
    r"""只转换 \mathcal, \mathbb, \mathbf 为 Unicode"""

    def replace_math_font(match):
        full_match = match.group(0)
        try:
            return unicodeit.replace(full_match)
        except:
            return full_match

    pattern = r'\\math(?:cal|bb|bf)\{[^}]+\}'
    text = re.sub(pattern, replace_math_font, text)

    return text


def add_spaces_around_display_math(text):
    """在 $$...$$ 整体前后添加空格"""

    # 匹配整个 $$...$$ 块
    # 在前面添加空格(如果前面不是空白字符或行首)
    # 在后面添加空格(如果后面不是空白字符或行尾)
    def replace_with_spaces(match):
        full_match = match.group(0)  # 完整的 $$...$$
        prefix = match.group(1)       # 前面的字符(如果有)
        content = match.group(2)      # $$ 之间的内容
        suffix = match.group(3)       # 后面的字符(如果有)

        result = ''
        if prefix:
            result += prefix + ' '
        result += '$$' + content + '$$'
        if suffix:
            result += ' ' + suffix

        return result

    # 匹配模式:
    # ([^\s\n])? - 可选的非空白字符(前面的字符)
    # \$\$(.*?)\$\$ - $$...$$ 块
    # ([^\s\n])? - 可选的非空白字符(后面的字符)
    pattern = r'([^\s\n])?\$\$(.*?)\$\$([^\s\n])?'
    text = re.sub(pattern, replace_with_spaces, text, flags=re.DOTALL)

    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="README.md")
    args = parser.parse_args()

    with open(args.file, 'r', encoding='utf-8') as f:
        content = f.read()

    content = add_spaces_around_display_math(content)
    converted = convert_math_fonts_to_unicode(content)

    os.makedirs(".cache", exist_ok=True)
    target_file = os.path.join(".cache", args.file)
    shutil.copy2(args.file, target_file)
    with open(args.file, 'w', encoding='utf-8') as f:
        f.write(converted)
