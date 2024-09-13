import ITextService from "../model/textService";
import ITextStore from "../model/textStore";
import textStore from "../store/textStore";


class TextService implements ITextService{
    
    private static instance: TextService;
    private textStore: ITextStore;

    private constructor(textStore:ITextStore) {
        this.textStore = textStore;        
    }

    public static getInstance(themeStore:ITextStore): TextService {
        if (!TextService.instance) {
            TextService.instance = new TextService(themeStore);
        }
        return TextService.instance;
    }

    public toggle(): void {
        this.textStore.setText(textStore.getOutput(), "");
    }

    public setInput(input:string): void {
        this.textStore.setInput(input);
    }

    public setOutput(output:string): void {
        this.textStore.setOutput(output);
    }

}

export default TextService.getInstance(textStore);