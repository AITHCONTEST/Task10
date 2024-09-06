
interface ILangService{
    toggleLang(): void;
    getNameByCode(code: string):string;
}

export default ILangService;