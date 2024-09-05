import { makeAutoObservable } from "mobx";
import ILangStore from "../model/langStore";


class LangStore implements ILangStore{   

    private static instance: LangStore;

    public fromLang: string;
    public toLang: string;

    private constructor() {
        this.fromLang = "ru";
        this.toLang = "mans";

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
    }

}

export default LangStore.getInstance();