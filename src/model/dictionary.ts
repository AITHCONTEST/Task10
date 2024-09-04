
type ZodiacSign = {
  name: string;
  englishName: string;
  dateRange: string;
};

interface IZodiacInfo {
  about: string;
  signs: ZodiacSign[];
}

interface IDictionary{
  [key: string]:IZodiacInfo;
}

export default IDictionary;