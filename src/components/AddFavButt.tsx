import star from "./../assets/new/star.svg";
import gstar from "./../assets/new/gold_star.svg";
import Button from "./Button";
import styles from "./styles/buttons/addFav.module.scss";
import favsStore from "../store/favsStore";
import textStore from "../store/textStore";
import { useEffect, useState } from "react";

const AddFavButt = () => {
    const [isActive, setIsActive] = useState(false);

    useEffect(()=>{
        const text = textStore.getInput();
        if(favsStore.exists(text)){
            setIsActive(true);
        }else{
            setIsActive(false);
        }
    }, [textStore.input])

    const tap = () => {
        setIsActive(!isActive);
        const text = textStore.getInput();
        if(favsStore.exists(text)){
            favsStore.remove(text)
        }else{
            favsStore.add(text);
        }
    }
    const dynamicStyle = {
        transform: isActive ? 'rotateY(900deg)' : 'rotateY(0deg)',
        
    };

    return (
        <div className={styles.addFav} style={dynamicStyle}>
            <Button img={isActive ? gstar : star} tap={tap}  alt={"add"}/>
        </div>
        
    );
};

export default AddFavButt;
