import { makeAutoObservable } from "mobx";
import ILangStore from "../model/langStore";


class LangStore implements ILangStore{   

    private static instance: LangStore;

    public fromLang: string;
    public toLang: string;

    private constructor() {
        this.fromLang = "";
        this.toLang = "";
        this.loadFromLocalStorage();
        
        makeAutoObservable(this);
    }

    public static getInstance(): LangStore {
        if (!LangStore.instance) {
            LangStore.instance = new LangStore();
        }
        return LangStore.instance;
    }

    public getFromlang(): string {
        return this.fromLang;
    }

    public getTolang(): string {
        return this.toLang;
    }

    public setLangs(from: string, to: string): void {
        this.fromLang = from;
        this.toLang = to;
        this.saveToLocalStorage();
    }

    private saveToLocalStorage(): void {
        localStorage.setItem('fromLang', this.fromLang);
        localStorage.setItem('toLang', this.toLang);
    }

    private loadFromLocalStorage(): boolean {
        const fromLang = localStorage.getItem('fromLang');
        if (fromLang) {
            this.fromLang = fromLang;
        }
        const toLang = localStorage.getItem('toLang');
        if (toLang) {
            this.toLang = toLang;
        }
        return !!fromLang && !!toLang;
    }


}

export default LangStore.getInstance();