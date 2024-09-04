
interface ILangStore {    
    getFromlang(): string;
    getTolang(): string;
    setLangs(from: string, to: string): void;
}

export default ILangStore;