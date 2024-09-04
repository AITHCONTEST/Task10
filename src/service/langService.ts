import ILangService from "../model/langService";
import ILangStore from "../model/langStore";
import langStore from "../store/langStore";


class LangService implements ILangService{
    
    private static instance: LangService;
    private langStore: ILangStore;

    private constructor(langStore:ILangStore) {
        this.langStore = langStore;
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

}

export default LangService.getInstance(langStore);