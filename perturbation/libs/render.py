from dataclasses import dataclass
import numpy as np

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

# Lastest Pillow (PIL) support complex text script layout, e.g., based on the Harfbuzz shaping engine
# https://pillow.readthedocs.io/en/stable/releasenotes/4.2.0.html?highlight=harfbuzz#added-complex-text-rendering
# Pillow 9.1.0 has been tested in this work.

class TextImage:
    def __init__(self, text: str, font: str, size=10, language=None, margin=0):
        self.text = text
        self.font = font # name or path
        self.size = size
        self.language = language
        self.init_config(margin=margin)

    def init_config(self, margin=0):
        # # static size
        # self.configs = { # default configuration
        #     'x': self.size,
        #     'y': self.size*2,
        #     'width': self.size*(len(self.text)+1), # add 1 because x = self.size
        #     'height': self.size*5,
        #     'background': 0, # black background
        #     'fill': 255, # white text
        # }

        # dynamic size
        try:
            left, top, right, bottom = ImageFont.truetype(f'{self.font}', self.size).getbbox(self.text)
        except:
            left, top, right, bottom = ImageFont.truetype(f'{self.font}.ttf', self.size).getbbox(self.text)
        top = abs(min(top, 0))
        left = abs(min(left, 0))
        self.configs = { # default configuration
            'x': left + margin,
            'y': top + margin,
            'width': right + left + 2 * margin,
            'height': bottom + top + 2 * margin,
            'background': 0, # black background
            'fill': 255, # white text
        }

    def draw(self) -> Image:
        try:
            font = ImageFont.truetype(f'{self.font}', self.size)
        except:
            font = ImageFont.truetype(f'{self.font}.ttf', self.size)
        img = Image.new('L', (self.configs['width'], self.configs['height']), self.configs['background']) # black background
        draw = ImageDraw.Draw(img)
        draw.text(
            (self.configs['x'], self.configs['y']), # left spacing and vertically middle
            self.text,
            fill=self.configs['fill'],
            font=font,
            language=self.language,
        )

        assert(not (np.asarray(img)==self.configs['background']).all()), 'The image for {} is empty {}.'.format(self.text, self.configs)
        return img

    def get_empty_offset(self):
        def _top_empty_offset(img):
            row = img.shape[0]
            for i in range(row):
                if not (img[i]==self.configs['background']).all():
                    return i + 1
            return 0

        img = np.asarray(self.draw())
        top_offset = _top_empty_offset(img)
        left_offset = _top_empty_offset(img.T)
        bottom_offset = _top_empty_offset(np.flip(img, 0))
        right_offset = _top_empty_offset(np.flip(img.T, 0))
        return left_offset, right_offset, top_offset, bottom_offset

    def get_np(self, strip=False, binarize=False) -> np.ndarray:
        def _strip_top(img):
            row = img.shape[0]
            for i in range(row):
                if not (img[i]==self.configs['background']).all():
                    img = img[i:]
                    row -= i
                    break
            return img

        img = np.asarray(self.draw())

        if strip:
            img = _strip_top(img)
            img = _strip_top(img.T).T
            img = np.flip(_strip_top(np.flip(img, 0)), 0)
            img = np.flip(_strip_top(np.flip(img.T, 0)), 0).T

        if binarize:
            img = (img != self.configs['background']).astype(int) # not black

        return img, Image.fromarray(np.uint8(img * (self.configs['fill'] if binarize else 1)))

    def get_vec(self, strip=False, binarize=False) -> np.ndarray:
        img = self.get_np(strip, binarize)[0]
        return img.reshape(-1)

@dataclass
class TextObject:
    text: str
    frequency: int
    main_char_id: int = 0
    textimage: TextImage = None

    @classmethod
    def from_line(cls, line: str, freq: int = 0, main_char_id: int = 0, font: str = None, size=10, language=None):
        return cls(line, freq, main_char_id, TextImage(line, font, size, language) if font is not None else None)

    def set_font(self, font: str, size=None):
        configs = self.textimage.configs
        self.textimage = TextImage(
            self.text, 
            font, 
            self.size if size is None else size,
            self.textimage.language
        )
        self.textimage.configs = configs

    def pretty(self):
        return '{}:{}'.format(self.text, self.frequency)

    def __len__(self):
        return len(self.text)

    def similary(self, other):
        raise NotImplementedError()
        # assert(isinstance(other, TextObject))
        # return len(set(self.text) & set(other.text)) * 2 / (len(self) + len(other))

    def get_glyph_vec(self, strip=False, binarize=False):
        assert(self.textimage is not None), 'Please set font for the text.'
        return self.textimage.get_vec(strip=strip, binarize=binarize)

    def crop_images_in_textobjs(textobjs, max_width=1000, max_height=1000):
        _textobjs = [
            textobj 
            for textobj in textobjs
            if textobj.textimage.configs['width'] <= max_width and textobj.textimage.configs['height'] <= max_height
        ]

        print('{} textobjs have been skipped. Width > {}, Height > {}'.format(len(textobjs) - len(_textobjs), max_width, max_height))
        textobjs = _textobjs

        max_width, max_height = 0, 0

        for textobj in textobjs:
            if textobj.textimage.configs['width'] > max_width:
                max_width = textobj.textimage.configs['width']
            if textobj.textimage.configs['height'] > max_height:
                max_height = textobj.textimage.configs['height']

        # re-configuration
        for textobj in textobjs:
            textobj.textimage.configs['x'] = (max_width - textobj.textimage.configs['width']) / 2
            textobj.textimage.configs['y'] = (max_height - textobj.textimage.configs['height']) / 2
            textobj.textimage.configs['width'] = max_width
            textobj.textimage.configs['height'] = max_height

        return textobjs, {
            'max_width':max_width,
            'max_height':max_height
        }