import ILangService from "../model/langService";
import ILangStore from "../model/langStore";
import langStore from "../store/langStore";


class LangService implements ILangService{
    
    private static instance: LangService;
    private langStore: ILangStore;
    private langs:{[key:string]:string};

    private constructor(langStore:ILangStore) {
        this.langStore = langStore;
        this.langs = {
            ru: "Русский",
            mans: "Мансийский"
        }
        if(!langStore.getTolang() || !langStore.getFromlang()){
            this.setLangs("ru", "mans")
        }
        
    }

    public static getInstance(themeStore:ILangStore): LangService {
        if (!LangService.instance) {
            LangService.instance = new LangService(themeStore);
        }
        return LangService.instance;
    }

    public toggleLang(): void {
        this.langStore.setLangs(langStore.getTolang(), langStore.getFromlang());
    }

    private setLangs(fromLang:string, toLang:string): void {
        this.langStore.setLangs(fromLang, toLang);
    }

    public getNameByCode(code: string):string{
        return this.langs[code];
    }

}

export default LangService.getInstance(langStore);